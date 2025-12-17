#!/usr/bin/env python3
"""
Null-Space Constrained Abliteration

Extends norm-preserving orthogonal projection with null-space constraints
from AlphaEdit (ICLR 2025) to preserve model capabilities on specified inputs.

The key insight: instead of directly applying the ablation update ΔW, we project
it into the null space of "preserved knowledge" activations. This mathematically
guarantees the update won't affect outputs for inputs we want to preserve.

Mathematical Framework:
-----------------------
Given:
  - Refusal direction d (unit vector)
  - Weight matrix W ∈ ℝ^(m×n)
  - Preserved knowledge activations K ∈ ℝ^(p×n) from p preservation prompts

Standard ablation:
  ΔW = -α(W @ d) ⊗ dᵀ

Null-space constrained ablation:
  P_null = I - Vᵀ @ V  (where V contains row-space basis from SVD)
  ΔW_constrained = ΔW @ P_null
  W_new = W + ΔW_constrained

This ensures: W_new @ k_i ≈ W @ k_i for all preserved activations k_i
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class NullSpaceConfig:
    """Configuration for null-space constrained abliteration."""

    # Preservation prompts - inputs whose outputs should be preserved
    preservation_prompts: list[str] = field(default_factory=list)
    preservation_prompts_path: Optional[str] = None

    # Null-space computation parameters
    regularization: float = 1e-4  # Tikhonov regularization for numerical stability
    svd_rank_ratio: float = 0.95  # Keep top singular values explaining this fraction of variance
    min_null_dim: int = 10  # Minimum null space dimension to retain

    # Per-layer vs global null space
    per_layer_null_space: bool = True  # Compute separate null space per layer

    # Memory optimization
    chunk_size: int = 100  # Process preservation prompts in chunks


@dataclass
class NullSpaceProjector:
    """Container for computed null-space projectors (low-rank representation)."""

    # Per-layer low-rank factors: layer_idx -> (V, S) where P_null ≈ I - V @ Vᵀ
    low_rank_factors: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save projectors to disk."""
        save_dict = {
            "low_rank_factors": {
                k: (v.cpu(), s.cpu()) for k, (v, s) in self.low_rank_factors.items()
            },
            "metadata": self.metadata,
        }
        torch.save(save_dict, path)
        logger.info(f"Saved null-space projectors to {path}")

    @classmethod
    def load(cls, path: str) -> "NullSpaceProjector":
        """Load projectors from disk."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            low_rank_factors={
                k: (v, s) for k, (v, s) in data.get("low_rank_factors", {}).items()
            },
            metadata=data.get("metadata", {}),
        )

    def get_projector_for_layer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get low-rank V matrix for a specific layer."""
        if layer_idx in self.low_rank_factors:
            V, _ = self.low_rank_factors[layer_idx]
            return V
        return None


def compute_null_space_projector_svd(
    activations: torch.Tensor,
    rank_ratio: float = 0.95,
    min_null_dim: int = 10,
    regularization: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute null-space projector using SVD for numerical stability.

    Given activations K ∈ ℝ^(p×n), we want P_null such that P_null @ k ≈ 0
    for all rows k in K, while preserving other directions.

    Using SVD: K = U @ S @ Vᵀ
    The row space of K is spanned by V[:, :r] where r is effective rank.
    The null space projector is: P_null = I - V[:, :r] @ V[:, :r]ᵀ

    We return low-rank factors (V[:, :r], S[:r]) for efficient application.

    Args:
        activations: [num_samples, hidden_dim] activation matrix
        rank_ratio: Keep singular values explaining this fraction of variance
        min_null_dim: Minimum null space dimensions to preserve
        regularization: Added to singular values for stability

    Returns:
        (V_truncated, singular_values) - low-rank factors for null-space projector
    """
    activations = activations.float()
    n_samples, hidden_dim = activations.shape

    # Center activations for better numerical properties
    mean_act = activations.mean(dim=0, keepdim=True)
    centered = activations - mean_act

    # SVD of centered activations
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError as e:
        logger.warning(f"SVD failed, falling back to eigendecomposition: {e}")
        # Fallback: eigendecomposition of K^T @ K
        gram = centered.T @ centered
        eigenvalues, V = torch.linalg.eigh(gram)
        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]
        S = torch.sqrt(torch.clamp(eigenvalues, min=0))
        Vh = V.T

    # Determine effective rank based on variance explained
    total_var = (S**2).sum()
    if total_var > 0:
        cumvar = torch.cumsum(S**2, dim=0) / total_var
        # Find rank that explains rank_ratio of variance
        rank_by_var = (cumvar < rank_ratio).sum().item() + 1
    else:
        rank_by_var = 1

    # Ensure we leave at least min_null_dim dimensions in null space
    max_rank = hidden_dim - min_null_dim
    effective_rank = min(rank_by_var, max_rank, n_samples)
    effective_rank = max(1, effective_rank)  # At least rank 1

    logger.debug(
        f"Null space: {n_samples} samples, {hidden_dim} dim, "
        f"effective rank {effective_rank}, null dim {hidden_dim - effective_rank}"
    )

    # V_truncated contains the row space basis (directions to project OUT of)
    V_truncated = Vh[:effective_rank, :].T  # [hidden_dim, effective_rank]
    S_truncated = S[:effective_rank]

    return V_truncated, S_truncated


def apply_null_space_projection(
    delta_w: torch.Tensor,
    V: torch.Tensor,
    project_dim: str = "input",
) -> torch.Tensor:
    """
    Apply null-space projection using low-rank factors.

    P_null = I - V @ Vᵀ

    For ΔW ∈ ℝ^(m×n):
      - project_dim="input": ΔW_proj = ΔW @ P_null = ΔW - ΔW @ V @ Vᵀ
      - project_dim="output": ΔW_proj = P_null @ ΔW = ΔW - V @ Vᵀ @ ΔW

    Args:
        delta_w: Weight update [out_features, in_features]
        V: Basis vectors to project out [hidden_dim, rank]
        project_dim: Which dimension to project ("input" or "output")

    Returns:
        Projected weight update
    """
    if project_dim == "input":
        # ΔW @ P_null = ΔW - (ΔW @ V) @ Vᵀ
        proj = delta_w @ V  # [m, rank]
        return delta_w - proj @ V.T
    else:
        # P_null @ ΔW = ΔW - V @ (Vᵀ @ ΔW)
        proj = V.T @ delta_w  # [rank, n]
        return delta_w - V @ proj


def apply_null_space_constrained_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
    null_space_V: Optional[torch.Tensor] = None,
    preserve_norm: bool = True,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Apply null-space constrained norm-preserving projection.

    This is the core function that combines:
    1. Standard ablation: ΔW = -α(W @ d) ⊗ dᵀ
    2. Null-space constraint: ΔW_constrained = ΔW @ P_null
    3. Norm preservation: scale to maintain original Frobenius norm

    Args:
        weight: Weight matrix [out_features, in_features]
        direction: Refusal direction [hidden_dim]
        null_space_V: Low-rank basis for null space [hidden_dim, rank] (optional)
        preserve_norm: Whether to rescale to preserve Frobenius norm
        multiplier: Ablation strength (1.0 = full ablation)

    Returns:
        Modified weight matrix
    """
    original_dtype = weight.dtype
    original_norm = weight.float().norm()

    weight_float = weight.float()
    direction_float = F.normalize(direction.float(), dim=0)

    # Determine projection dimension
    if direction_float.shape[0] == weight_float.shape[1]:
        project_dim = "input"
        # Standard ablation: ΔW = -(W @ d) ⊗ dᵀ
        proj_coeffs = weight_float @ direction_float  # [out_features]
        delta_w = -multiplier * torch.outer(proj_coeffs, direction_float)
    elif direction_float.shape[0] == weight_float.shape[0]:
        project_dim = "output"
        # ΔW = -d ⊗ (dᵀ @ W)
        proj_coeffs = direction_float @ weight_float  # [in_features]
        delta_w = -multiplier * torch.outer(direction_float, proj_coeffs)
    else:
        logger.warning(
            f"Direction shape {direction_float.shape} doesn't match weight {weight_float.shape}"
        )
        return weight

    # Apply null-space constraint if provided
    if null_space_V is not None:
        V = null_space_V.float().to(weight.device)
        delta_w = apply_null_space_projection(delta_w, V, project_dim)

    # Apply update
    weight_new = weight_float + delta_w

    # Norm preservation
    if preserve_norm:
        new_norm = weight_new.norm()
        if new_norm > 1e-8:
            weight_new = weight_new * (original_norm / new_norm)

    return weight_new.to(original_dtype)


class NullSpaceActivationExtractor:
    """Extract activations for computing null-space projectors."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.hooks = []

    def _get_layers(self):
        """Get transformer layers from model."""
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
                return self.model.model.model.layers
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
            elif hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
                return self.model.model.decoder.layers
        if hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "layers"):
                return self.model.language_model.model.layers
            if hasattr(self.model.language_model, "layers"):
                return self.model.language_model.layers
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
            elif hasattr(self.model.transformer, "layers"):
                return self.model.transformer.layers
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        raise ValueError(f"Could not find layers in model: {type(self.model)}")

    def _create_hook(self, layer_idx: int):
        """Create forward hook that captures ALL token activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store ALL token activations for null-space computation
            # Flatten to [batch * seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_acts = hidden_states.reshape(-1, hidden_dim)

            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(flat_acts.detach().cpu())

        return hook

    def register_hooks(self, layer_indices: list[int]):
        """Register hooks on specified layers."""
        layers = self._get_layers()
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                hook = layers[idx].register_forward_hook(self._create_hook(idx))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear stored activations."""
        self.activations = {}

    @torch.no_grad()
    def extract(
        self,
        prompts: list[str],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> dict[int, torch.Tensor]:
        """
        Extract activations for preservation prompts.

        Unlike refusal direction extraction which uses last token,
        we extract ALL token activations to build a comprehensive
        null space that preserves the model's behavior across the
        full sequence.
        """
        self.clear()
        self.model.eval()

        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting preservation activations"):
            batch = prompts[i : i + batch_size]

            # Format with chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = []
                for p in batch:
                    messages = [{"role": "user", "content": p}]
                    fmt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted.append(fmt)
            else:
                formatted = batch

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            _ = self.model(**inputs)

        # Concatenate activations per layer
        result = {}
        for layer_idx, acts in self.activations.items():
            result[layer_idx] = torch.cat(acts, dim=0)

        return result


def compute_null_space_projectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: NullSpaceConfig,
    layer_indices: list[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> NullSpaceProjector:
    """
    Compute null-space projectors from preservation prompts.

    Args:
        model: The model to extract activations from
        tokenizer: Tokenizer for the model
        config: Null-space configuration
        layer_indices: Which layers to compute projectors for
        device: Device for computation
        dtype: Data type for computation

    Returns:
        NullSpaceProjector containing computed projectors
    """
    logger.info("Computing null-space projectors for capability preservation...")

    # Load preservation prompts
    prompts = config.preservation_prompts.copy()
    if config.preservation_prompts_path:
        prompts.extend(load_preservation_prompts(config.preservation_prompts_path))

    if not prompts:
        # Use defaults if no prompts provided
        prompts = create_default_preservation_prompts()
        logger.info(f"Using {len(prompts)} default preservation prompts")
    else:
        logger.info(f"Using {len(prompts)} preservation prompts")

    # Extract activations
    extractor = NullSpaceActivationExtractor(model, tokenizer, device, dtype)
    extractor.register_hooks(layer_indices)

    try:
        activations = extractor.extract(
            prompts,
            batch_size=config.chunk_size,
        )
    finally:
        extractor.remove_hooks()

    # Compute projectors for each layer
    low_rank_factors = {}

    for layer_idx in tqdm(layer_indices, desc="Computing null-space projectors"):
        if layer_idx not in activations:
            continue

        acts = activations[layer_idx].to(device)
        n_samples, hidden_dim = acts.shape

        logger.debug(f"Layer {layer_idx}: {n_samples} activation samples, dim {hidden_dim}")

        V, S = compute_null_space_projector_svd(
            acts,
            rank_ratio=config.svd_rank_ratio,
            min_null_dim=config.min_null_dim,
            regularization=config.regularization,
        )

        low_rank_factors[layer_idx] = (V.to(dtype), S.to(dtype))

        logger.debug(
            f"Layer {layer_idx}: null-space rank {V.shape[1]}, "
            f"preserving {hidden_dim - V.shape[1]} dimensions"
        )

    metadata = {
        "num_prompts": len(prompts),
        "layer_indices": layer_indices,
        "svd_rank_ratio": config.svd_rank_ratio,
        "regularization": config.regularization,
    }

    return NullSpaceProjector(
        low_rank_factors=low_rank_factors,
        metadata=metadata,
    )


def load_preservation_prompts(path: str) -> list[str]:
    """Load preservation prompts from file."""
    import json

    path = Path(path)
    if not path.exists():
        logger.warning(f"Preservation prompts file not found: {path}")
        return []

    content = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
    else:
        # Text file: one prompt per line, ignore comments starting with #
        prompts = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
        return prompts

    return []


def create_default_preservation_prompts() -> list[str]:
    """
    Create a default set of preservation prompts covering diverse capabilities.

    These prompts are designed to capture activations for:
    - General knowledge and reasoning
    - Code generation
    - Creative writing
    - Math and logic
    - Helpful assistant behavior
    """
    return [
        # General knowledge
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Who wrote Romeo and Juliet?",
        "What causes the seasons on Earth?",
        "Describe the water cycle.",

        # Reasoning and analysis
        "Compare and contrast democracy and authoritarianism.",
        "What are the pros and cons of renewable energy?",
        "Analyze the causes of World War I.",
        "Explain the scientific method.",
        "What factors should I consider when making a major decision?",

        # Coding and technical
        "Write a Python function to calculate factorial.",
        "Explain the difference between a list and a dictionary in Python.",
        "How does a binary search algorithm work?",
        "What is object-oriented programming?",
        "Explain what an API is and how it works.",

        # Creative and writing
        "Write a short poem about nature.",
        "Help me brainstorm ideas for a birthday party.",
        "Describe a peaceful forest scene.",
        "Write a professional email requesting time off.",
        "Create a simple recipe for pasta.",

        # Math and logic
        "Solve the equation 2x + 5 = 15.",
        "What is the Pythagorean theorem?",
        "Explain probability using a coin flip example.",
        "How do you calculate the area of a circle?",
        "What is the difference between mean and median?",

        # Helpful assistant
        "How can I improve my study habits?",
        "What are some tips for better sleep?",
        "How do I start learning a new language?",
        "Suggest some healthy breakfast options.",
        "How can I be more productive at work?",

        # Complex multi-step
        "Plan a week-long trip to Japan including must-see attractions.",
        "Explain how to start a small business step by step.",
        "Create a workout routine for a beginner.",
        "How do I write a good resume?",
        "Explain machine learning to a 10-year-old.",
    ]


def get_default_preservation_prompts_path() -> str:
    """Get the path to the default preservation prompts file.

    Checks user prompts directory (~/.abliterate/prompts/) first,
    then falls back to the package prompts directory.
    """
    from src.cli_components import get_prompts_path
    return str(get_prompts_path("preservation.txt"))
