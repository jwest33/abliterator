#!/usr/bin/env python3
"""
Norm-Preserving Orthogonal Projection Abliteration for HuggingFace Models (Safetensors)

This script removes refusal behavior from language models by:
1. Computing refusal directions from contrastive prompt pairs (harmful vs harmless)
2. Applying norm-preserving orthogonal projection to ablate refusal directions from all linear layers
3. Saving the modified model as a new safetensors model

The norm-preserving aspect ensures that after removing the refusal component,
the remaining weight matrix is rescaled to maintain its original Frobenius norm,
preserving the model's overall activation magnitudes.

Author: Jake (with Claude)
License: MIT
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Data Classes


@dataclass
class AbliterationConfig:
    """Configuration for the abliteration process."""

    model_path: str
    output_path: str
    harmful_prompts_path: str = "./prompts/harmful.txt"
    harmless_prompts_path: str = "./prompts/harmless.txt"
    num_prompts: Optional[int] = None  # Number of prompts to sample (None = use all)
    harmful_prompts: list[str] = field(default_factory=list)
    harmless_prompts: list[str] = field(default_factory=list)
    target_layers: Optional[list[int]] = None  # None = all layers
    extraction_layer_indices: Optional[list[int]] = None  # Layers to extract directions from
    use_mean_direction: bool = True  # Average directions across extraction layers
    normalize_directions: bool = True
    norm_preservation: bool = True  # Rescale weights to preserve Frobenius norm
    direction_multiplier: float = 1.0  # Scale the ablation strength
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32  # Computation dtype
    save_directions: bool = True  # Save computed refusal directions
    load_directions_path: Optional[str] = None  # Load pre-computed directions
    batch_size: int = 4
    max_new_tokens: int = 1  # We only need the first forward pass
    token_position: str = "last"  # Which token position to extract: "last", "mean", or int
    filter_harmful_prompts: bool = True  # Filter out harmful prompts that aren't actually refused
    refusal_test_max_tokens: int = 50  # Max tokens to generate when testing for refusal
    refusal_keywords: list[str] = field(default_factory=lambda: [
        "I cannot", "I can't", "I'm not able", "I am not able",
        "I won't", "I will not", "I'm unable", "I am unable",
        "Sorry", "I apologize", "As an AI", "As a language model",
        "I'm an AI", "I am an AI", "not appropriate", "against my",
        "ethical", "illegal", "harmful", "dangerous", "unethical",
        "I must refuse", "I have to refuse", "cannot assist", "can't assist",
        "cannot help", "can't help", "cannot provide", "can't provide",
        "not able to help", "not able to assist", "not able to provide",
    ])


@dataclass
class RefusalDirections:
    """Container for computed refusal directions."""

    directions: dict[int, torch.Tensor]  # layer_idx -> direction vector
    mean_direction: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save directions to disk."""
        save_dict = {
            "directions": {k: v.cpu() for k, v in self.directions.items()},
            "mean_direction": self.mean_direction.cpu() if self.mean_direction is not None else None,
            "metadata": self.metadata,
        }
        torch.save(save_dict, path)
        logger.info(f"Saved refusal directions to {path}")

    @classmethod
    def load(cls, path: str) -> "RefusalDirections":
        """Load directions from disk."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            directions=data["directions"],
            mean_direction=data["mean_direction"],
            metadata=data.get("metadata", {}),
        )


# Activation Extraction


class ActivationExtractor:
    """Extracts hidden state activations from transformer layers."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: AbliterationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.hooks = []

    def _get_layers(self):
        """Get the transformer layers from the model."""
        # Handle different model architectures
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
            elif hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
                return self.model.model.decoder.layers
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
            elif hasattr(self.model.transformer, "layers"):
                return self.model.transformer.layers
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        raise ValueError(f"Could not find layers in model architecture: {type(self.model)}")

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Extract based on token position config
            if self.config.token_position == "last":
                # Get the last non-padding token
                extracted = hidden_states[:, -1, :]
            elif self.config.token_position == "mean":
                extracted = hidden_states.mean(dim=1)
            elif isinstance(self.config.token_position, int):
                extracted = hidden_states[:, self.config.token_position, :]
            else:
                extracted = hidden_states[:, -1, :]

            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(extracted.detach().cpu())

        return hook

    def register_hooks(self, layer_indices: Optional[list[int]] = None):
        """Register forward hooks on specified layers."""
        layers = self._get_layers()
        num_layers = len(layers)

        if layer_indices is None:
            # Default to middle-to-later layers where refusal is typically encoded
            layer_indices = list(range(num_layers // 4, 3 * num_layers // 4))

        for idx in layer_indices:
            if 0 <= idx < num_layers:
                hook = layers[idx].register_forward_hook(self._create_hook(idx))
                self.hooks.append(hook)

        logger.info(f"Registered hooks on layers: {layer_indices}")
        return layer_indices

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    @torch.no_grad()
    def extract_activations(self, prompts: list[str]) -> dict[int, torch.Tensor]:
        """Extract activations for a list of prompts."""
        self.clear_activations()
        self.model.eval()

        for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="Extracting activations"):
            batch_prompts = prompts[i : i + self.config.batch_size]

            # Tokenize with chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompts = []
                for prompt in batch_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
            else:
                formatted_prompts = batch_prompts

            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.config.device)

            # Forward pass to trigger hooks
            _ = self.model(**inputs)

        # Concatenate all activations for each layer
        result = {}
        for layer_idx, acts in self.activations.items():
            result[layer_idx] = torch.cat(acts, dim=0)

        return result


# Refusal Direction Computation


def compute_refusal_directions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
) -> RefusalDirections:
    """
    Compute refusal directions from contrastive prompt pairs.

    The refusal direction is computed as the mean difference between
    harmful and harmless prompt activations at each layer.
    """
    logger.info("Computing refusal directions from contrastive prompts...")

    extractor = ActivationExtractor(model, tokenizer, config)

    # Determine which layers to extract from
    layers = extractor._get_layers()
    num_layers = len(layers)

    if config.extraction_layer_indices is None:
        # Default: extract from middle-to-later layers
        extraction_layers = list(range(num_layers // 4, 3 * num_layers // 4))
    else:
        extraction_layers = config.extraction_layer_indices

    extractor.register_hooks(extraction_layers)

    try:
        # Extract activations for harmful prompts
        logger.info(f"Extracting activations for {len(config.harmful_prompts)} harmful prompts...")
        harmful_activations = extractor.extract_activations(config.harmful_prompts)

        extractor.clear_activations()

        # Extract activations for harmless prompts
        logger.info(f"Extracting activations for {len(config.harmless_prompts)} harmless prompts...")
        harmless_activations = extractor.extract_activations(config.harmless_prompts)

    finally:
        extractor.remove_hooks()

    # Compute refusal directions as mean difference
    directions = {}
    for layer_idx in extraction_layers:
        if layer_idx in harmful_activations and layer_idx in harmless_activations:
            harmful_mean = harmful_activations[layer_idx].mean(dim=0)
            harmless_mean = harmless_activations[layer_idx].mean(dim=0)

            # Refusal direction: harmful - harmless
            direction = harmful_mean - harmless_mean

            if config.normalize_directions:
                direction = F.normalize(direction, dim=0)

            directions[layer_idx] = direction.to(config.dtype)

            logger.debug(f"Layer {layer_idx}: direction norm = {direction.norm().item():.4f}")

    # Compute mean direction across all layers
    mean_direction = None
    if config.use_mean_direction and directions:
        stacked = torch.stack(list(directions.values()))
        mean_direction = stacked.mean(dim=0)
        if config.normalize_directions:
            mean_direction = F.normalize(mean_direction, dim=0)

    metadata = {
        "num_harmful_prompts": len(config.harmful_prompts),
        "num_harmless_prompts": len(config.harmless_prompts),
        "extraction_layers": extraction_layers,
        "normalized": config.normalize_directions,
        "token_position": config.token_position,
    }

    logger.info(f"Computed refusal directions for {len(directions)} layers")

    return RefusalDirections(directions=directions, mean_direction=mean_direction, metadata=metadata)


# Norm-Preserving Orthogonal Projection


def orthogonal_projection_matrix(direction: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal projection matrix that removes the component along `direction`.

    P = I - (d @ d^T) / (d^T @ d)

    For a normalized direction vector, this simplifies to:
    P = I - d @ d^T
    """
    direction = direction.view(-1, 1)  # Column vector
    # Ensure direction is normalized
    direction = direction / direction.norm()
    # Projection matrix: I - d @ d^T
    proj_matrix = torch.eye(direction.size(0), device=direction.device, dtype=direction.dtype)
    proj_matrix -= direction @ direction.T
    return proj_matrix


def apply_norm_preserving_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
    preserve_norm: bool = True,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Apply norm-preserving orthogonal projection to a weight matrix.

    This removes the component of each row (or column, depending on the weight's role)
    that aligns with the refusal direction, then optionally rescales to preserve
    the original Frobenius norm.

    Args:
        weight: The weight matrix to modify [out_features, in_features]
        direction: The refusal direction vector [hidden_size]
        preserve_norm: Whether to rescale weights to preserve Frobenius norm
        multiplier: Scale factor for ablation strength (1.0 = full ablation)

    Returns:
        Modified weight matrix
    """
    original_dtype = weight.dtype
    original_norm = weight.float().norm()

    # Work in float32 for numerical stability
    weight_float = weight.float()
    direction_float = direction.float()

    # Ensure direction is normalized
    direction_float = F.normalize(direction_float, dim=0)

    # Determine which dimension to project based on weight shape
    # For most linear layers: weight is [out_features, in_features]
    # The direction typically matches in_features (the hidden dimension being projected)

    if direction_float.shape[0] == weight_float.shape[1]:
        # Project along input dimension (columns)
        # W_new = W @ P where P is the projection matrix
        # Equivalent to: W_new = W - (W @ d) @ d^T
        proj_coeffs = weight_float @ direction_float  # [out_features]
        proj_coeffs = proj_coeffs * multiplier
        adjustment = torch.outer(proj_coeffs, direction_float)  # [out_features, in_features]
        weight_new = weight_float - adjustment

    elif direction_float.shape[0] == weight_float.shape[0]:
        # Project along output dimension (rows)
        # W_new = P @ W
        # Equivalent to: W_new = W - d @ (d^T @ W)
        proj_coeffs = direction_float @ weight_float  # [in_features]
        proj_coeffs = proj_coeffs * multiplier
        adjustment = torch.outer(direction_float, proj_coeffs)  # [out_features, in_features]
        weight_new = weight_float - adjustment

    else:
        logger.warning(
            f"Direction shape {direction_float.shape} doesn't match weight shape {weight_float.shape}, skipping"
        )
        return weight

    # Norm preservation: rescale to maintain original Frobenius norm
    if preserve_norm:
        new_norm = weight_new.norm()
        if new_norm > 1e-8:  # Avoid division by zero
            weight_new = weight_new * (original_norm / new_norm)

    return weight_new.to(original_dtype)


# Model Abliteration


def get_linear_layer_names(model: AutoModelForCausalLM) -> list[str]:
    """Get names of all linear layers in the model."""
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
    return linear_names


def get_layer_index_from_name(name: str) -> Optional[int]:
    """Extract layer index from a parameter name."""
    import re

    # Common patterns: layers.0., h.0., decoder.layers.0., etc.
    patterns = [
        r"layers\.(\d+)\.",
        r"h\.(\d+)\.",
        r"block\.(\d+)\.",
        r"decoder\.layers\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def abliterate_model(
    model: AutoModelForCausalLM,
    directions: RefusalDirections,
    config: AbliterationConfig,
) -> AutoModelForCausalLM:
    """
    Apply norm-preserving orthogonal projection abliteration to the model.

    This modifies the model's weights in-place to remove refusal behavior.
    """
    logger.info("Applying norm-preserving orthogonal projection abliteration...")

    linear_names = get_linear_layer_names(model)
    logger.info(f"Found {len(linear_names)} linear layers")

    # Get the direction to use
    if config.use_mean_direction and directions.mean_direction is not None:
        primary_direction = directions.mean_direction.to(config.device)
        logger.info("Using mean direction across layers")
    else:
        # Use per-layer directions
        primary_direction = None
        logger.info("Using per-layer directions")

    modified_count = 0
    skipped_count = 0

    for name in tqdm(linear_names, desc="Abliterating layers"):
        # Get the module
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)

        weight = module.weight.data

        # Determine which direction to use
        layer_idx = get_layer_index_from_name(name)

        if config.target_layers is not None and layer_idx is not None:
            if layer_idx not in config.target_layers:
                skipped_count += 1
                continue

        if primary_direction is not None:
            direction = primary_direction
        elif layer_idx is not None and layer_idx in directions.directions:
            direction = directions.directions[layer_idx].to(config.device)
        elif directions.mean_direction is not None:
            direction = directions.mean_direction.to(config.device)
        else:
            logger.warning(f"No direction available for {name}, skipping")
            skipped_count += 1
            continue

        # Check if direction dimension matches either weight dimension
        if direction.shape[0] not in weight.shape:
            skipped_count += 1
            continue

        # Apply projection
        try:
            new_weight = apply_norm_preserving_projection(
                weight,
                direction,
                preserve_norm=config.norm_preservation,
                multiplier=config.direction_multiplier,
            )
            module.weight.data = new_weight
            modified_count += 1
        except Exception as e:
            logger.warning(f"Failed to abliterate {name}: {e}")
            skipped_count += 1

    logger.info(f"Modified {modified_count} layers, skipped {skipped_count} layers")

    return model


# Main Pipeline


def load_prompts_from_file(path: str, num_prompts: Optional[int] = None) -> list[str]:
    """Load prompts from a JSON or text file, optionally sampling randomly."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")
    else:
        # Assume text file with one prompt per line
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Randomly sample if num_prompts is specified and less than total
    if num_prompts is not None and num_prompts < len(prompts):
        prompts = random.sample(prompts, num_prompts)

    return prompts


# Refusal Detection and Filtering


def is_refusal_response(response: str, refusal_keywords: list[str]) -> bool:
    """
    Check if a model response indicates a refusal.

    Args:
        response: The model's generated response
        refusal_keywords: List of phrases that indicate refusal

    Returns:
        True if the response appears to be a refusal
    """
    response_lower = response.lower()
    for keyword in refusal_keywords:
        if keyword.lower() in response_lower:
            return True
    return False


@torch.no_grad()
def test_prompts_for_refusal_batch(
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
) -> list[tuple[str, bool, str]]:
    """
    Test multiple prompts for refusal in batches (much faster than one-by-one).

    Args:
        prompts: List of prompts to test
        model: The model to test
        tokenizer: The tokenizer
        config: Configuration with refusal keywords and generation settings

    Returns:
        List of (prompt, is_refused, response) tuples
    """
    model.eval()
    results = []

    # Process in batches
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i : i + config.batch_size]

        # Format with chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
        else:
            formatted_prompts = batch_prompts

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(config.device)

        # Generate responses for entire batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.refusal_test_max_tokens,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode each response in the batch
        for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # Get the input length for this specific prompt
            input_len = (inputs["attention_mask"][j] == 1).sum().item()
            generated_tokens = output[input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            is_refused = is_refusal_response(response, config.refusal_keywords)
            results.append((prompt, is_refused, response))

    return results


def filter_harmful_prompts_by_refusal(
    prompt_pool: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
    target_count: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """
    Filter harmful prompts to only include those that the model actually refuses.
    Uses batched generation for speed and samples additional prompts if needed
    to reach the target count.

    Args:
        prompt_pool: Full pool of available harmful prompts to sample from
        model: The model to test against
        tokenizer: The tokenizer
        config: Configuration settings
        target_count: Target number of refused prompts to collect (None = test all)

    Returns:
        Tuple of (refused_prompts, non_refused_prompts)
    """
    if target_count is None:
        target_count = len(prompt_pool)

    logger.info(f"Finding {target_count} refused prompts from pool of {len(prompt_pool)}...")

    refused_prompts = []
    non_refused_prompts = []
    tested_indices = set()

    # Shuffle indices for random sampling
    available_indices = list(range(len(prompt_pool)))
    random.shuffle(available_indices)

    with tqdm(total=target_count, desc="Finding refused prompts") as pbar:
        while len(refused_prompts) < target_count and len(tested_indices) < len(prompt_pool):
            # Calculate how many more we need, with some buffer for non-refusals
            needed = target_count - len(refused_prompts)
            # Sample more than needed to account for non-refusals (estimate ~80% refusal rate)
            sample_size = min(max(needed * 2, config.batch_size), len(prompt_pool) - len(tested_indices))

            # Get next batch of untested prompts
            batch_indices = []
            for idx in available_indices:
                if idx not in tested_indices:
                    batch_indices.append(idx)
                    if len(batch_indices) >= sample_size:
                        break

            if not batch_indices:
                break

            batch_prompts = [prompt_pool[idx] for idx in batch_indices]
            tested_indices.update(batch_indices)

            # Test batch
            batch_results = test_prompts_for_refusal_batch(batch_prompts, model, tokenizer, config)

            for prompt, is_refused, response in batch_results:
                if is_refused:
                    if len(refused_prompts) < target_count:
                        refused_prompts.append(prompt)
                        pbar.update(1)
                        logger.debug(f"REFUSED: {prompt[:50]}...")
                else:
                    non_refused_prompts.append(prompt)
                    logger.debug(f"NOT REFUSED: {prompt[:50]}... -> {response[:100]}...")

    logger.info(f"Refusal filtering complete:")
    logger.info(f"  - Prompts tested: {len(tested_indices)}")
    logger.info(f"  - Prompts refused: {len(refused_prompts)}")
    logger.info(f"  - Prompts NOT refused: {len(non_refused_prompts)}")

    if len(refused_prompts) < target_count:
        logger.warning(
            f"Could only find {len(refused_prompts)} refused prompts "
            f"(target was {target_count}) after testing all {len(prompt_pool)} prompts"
        )

    return refused_prompts, non_refused_prompts


def run_abliteration(config: AbliterationConfig):
    """Run the complete abliteration pipeline."""
    logger.info("=" * 60)
    logger.info("Norm-Preserving Orthogonal Projection Abliteration")
    logger.info("=" * 60)

    # Load model and tokenizer first (needed for refusal filtering)
    logger.info(f"Loading model from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    logger.info(f"Model loaded: {type(model).__name__}")

    # Load prompts from files
    # For harmful prompts with filtering: load ALL prompts as a pool, then filter to target count
    # For harmless prompts: sample directly since no filtering needed
    if config.filter_harmful_prompts and not config.load_directions_path:
        # Load full pool of harmful prompts for filtering
        harmful_prompt_pool = load_prompts_from_file(config.harmful_prompts_path, num_prompts=None)
        logger.info(f"Loaded {len(harmful_prompt_pool)} harmful prompts from {config.harmful_prompts_path}")

        # Filter to find target_count refused prompts
        target_count = config.num_prompts if config.num_prompts else len(harmful_prompt_pool)
        refused_prompts, _ = filter_harmful_prompts_by_refusal(
            harmful_prompt_pool, model, tokenizer, config, target_count=target_count
        )

        if len(refused_prompts) == 0:
            raise ValueError(
                "No harmful prompts were refused by the model! "
                "Cannot compute refusal directions without refused prompts. "
                "Either use different harmful prompts or disable filtering with --no_filter_prompts"
            )

        config.harmful_prompts = refused_prompts
        logger.info(f"Using {len(config.harmful_prompts)} refused prompts for refusal direction computation")
    else:
        # No filtering - just sample directly
        config.harmful_prompts = load_prompts_from_file(config.harmful_prompts_path, config.num_prompts)
        logger.info(f"Loaded {len(config.harmful_prompts)} harmful prompts from {config.harmful_prompts_path}")

    config.harmless_prompts = load_prompts_from_file(config.harmless_prompts_path, config.num_prompts)
    logger.info(f"Loaded {len(config.harmless_prompts)} harmless prompts from {config.harmless_prompts_path}")

    # Get or compute refusal directions
    if config.load_directions_path:
        logger.info(f"Loading pre-computed directions from {config.load_directions_path}")
        directions = RefusalDirections.load(config.load_directions_path)
    else:
        directions = compute_refusal_directions(model, tokenizer, config)

        if config.save_directions:
            directions_path = Path(config.output_path) / "refusal_directions.pt"
            directions_path.parent.mkdir(parents=True, exist_ok=True)
            directions.save(str(directions_path))

    # Apply abliteration
    model = abliterate_model(model, directions, config)

    # Save the modified model
    logger.info(f"Saving abliterated model to {config.output_path}...")
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Save config for reproducibility
    config_save = {
        "model_path": config.model_path,
        "target_layers": config.target_layers,
        "extraction_layer_indices": config.extraction_layer_indices,
        "use_mean_direction": config.use_mean_direction,
        "normalize_directions": config.normalize_directions,
        "norm_preservation": config.norm_preservation,
        "direction_multiplier": config.direction_multiplier,
        "token_position": config.token_position,
        "num_harmful_prompts": len(config.harmful_prompts),
        "num_harmless_prompts": len(config.harmless_prompts),
    }
    with open(output_path / "abliteration_config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    logger.info("=" * 60)
    logger.info("Abliteration complete!")
    logger.info(f"Output saved to: {config.output_path}")
    logger.info("=" * 60)

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Norm-Preserving Orthogonal Projection Abliteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses ./prompts/harmful.txt and ./prompts/harmless.txt)
  python abliterate.py --model_path ./my_model --output_path ./abliterated_model

  # Sample 100 random prompts from each file
  python abliterate.py --model_path ./my_model --output_path ./abliterated_model --num_prompts 100

  # With custom prompt files
  python abliterate.py \\
    --model_path ./my_model \\
    --output_path ./abliterated_model \\
    --harmful_prompts /path/to/harmful.json \\
    --harmless_prompts /path/to/harmless.json

  # Target specific layers only
  python abliterate.py \\
    --model_path ./my_model \\
    --output_path ./abliterated_model \\
    --target_layers 10 11 12 13 14 15
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the input model (HuggingFace format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the abliterated model",
    )
    parser.add_argument(
        "--harmful_prompts",
        type=str,
        default="./prompts/harmful.txt",
        help="Path to JSON or text file with harmful prompts (default: ./prompts/harmful.txt)",
    )
    parser.add_argument(
        "--harmless_prompts",
        type=str,
        default="./prompts/harmless.txt",
        help="Path to JSON or text file with harmless prompts (default: ./prompts/harmless.txt)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=30,
        help="Number of prompts to randomly sample from each file (default: use all)",
    )
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to abliterate (default: all layers)",
    )
    parser.add_argument(
        "--extraction_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to extract directions from (default: middle layers)",
    )
    parser.add_argument(
        "--direction_multiplier",
        type=float,
        default=1.0,
        help="Scale factor for ablation strength (default: 1.0)",
    )
    parser.add_argument(
        "--no_norm_preservation",
        action="store_true",
        help="Disable norm preservation (not recommended)",
    )
    parser.add_argument(
        "--per_layer_directions",
        action="store_true",
        help="Use per-layer directions instead of mean direction",
    )
    parser.add_argument(
        "--load_directions",
        type=str,
        default=None,
        help="Load pre-computed refusal directions from file",
    )
    parser.add_argument(
        "--no_save_directions",
        action="store_true",
        help="Don't save computed refusal directions",
    )
    parser.add_argument(
        "--no_filter_prompts",
        action="store_true",
        help="Disable filtering of harmful prompts (use all prompts even if not refused)",
    )
    parser.add_argument(
        "--refusal_test_tokens",
        type=int,
        default=50,
        help="Max tokens to generate when testing for refusal (default: 50)",
    )
    parser.add_argument(
        "--token_position",
        type=str,
        default="last",
        help="Token position for activation extraction: 'last', 'mean', or integer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Parse token position
    token_pos = args.token_position
    if token_pos.isdigit():
        token_pos = int(token_pos)

    config = AbliterationConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        harmful_prompts_path=args.harmful_prompts,
        harmless_prompts_path=args.harmless_prompts,
        num_prompts=args.num_prompts,
        target_layers=args.target_layers,
        extraction_layer_indices=args.extraction_layers,
        use_mean_direction=not args.per_layer_directions,
        norm_preservation=not args.no_norm_preservation,
        direction_multiplier=args.direction_multiplier,
        device=args.device,
        dtype=dtype_map[args.dtype],
        save_directions=not args.no_save_directions,
        load_directions_path=args.load_directions,
        batch_size=args.batch_size,
        token_position=token_pos,
        filter_harmful_prompts=not args.no_filter_prompts,
        refusal_test_max_tokens=args.refusal_test_tokens,
    )

    run_abliteration(config)


if __name__ == "__main__":
    main()
