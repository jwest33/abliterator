"""Jacobian Lens (J-lens) — computes J-space concept vectors via vector-Jacobian
products of final-layer logits w.r.t. intermediate hidden states.

Based on "Verbalizable Representations Form a Global Workspace in Language Models"
(transformer-circuits.pub 2026). For each concept token c and layer ℓ, computes
    g_{ℓ,c} = E_prompts [ ∂ logit_c(last position) / ∂ h_{ℓ, last position} ]
averaged over prompts via Welford's algorithm. These per-layer per-concept
vectors serve multiple downstream uses:
  - Per-layer refusal signal strength -> layer_target_map.json
  - Orthonormal basis for J-space-restricted refusal directions
  - Concept probes for workspace-based refusal detection
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.abliterate import (
    WelfordMeanAccumulator,
    get_transformer_layers,
)

logger = logging.getLogger(__name__)


DEFAULT_REFUSAL_CONCEPTS = [
    " cannot",
    " can",
    " sorry",
    " apolog",
    " unable",
    " refuse",
    "I",
]
DEFAULT_CONTRAST_CONCEPTS = [
    " Sure",
    " Here",
    " Of",
    " Yes",
]


@dataclass
class JLensConfig:
    """Configuration for J-lens vector extraction."""

    concepts: list[str] = field(default_factory=lambda: list(DEFAULT_REFUSAL_CONCEPTS))
    contrast_concepts: list[str] = field(default_factory=lambda: list(DEFAULT_CONTRAST_CONCEPTS))
    num_prompts: int = 32
    max_seq_len: int = 64
    batch_size: int = 2
    gradient_checkpointing: bool = False
    layer_indices: Optional[list[int]] = None  # None = all layers
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    # Basis construction
    basis_rank: Optional[int] = None  # None = keep all linearly independent concept dirs
    # Target-map generation defaults
    exclude_threshold: float = 0.2  # signal below this -> exclude layer
    min_multiplier: float = 0.1
    aggressive_threshold: float = 0.8


@dataclass
class JLensVectors:
    """Container for J-lens concept vectors and derived per-layer artifacts."""

    concept_vectors: dict[int, dict[str, torch.Tensor]]  # layer -> concept -> [hidden_dim]
    layer_signal: dict[int, float]                        # layer -> normalized signal in [0,1]
    basis: dict[int, torch.Tensor]                        # layer -> [hidden_dim, k]
    metadata: dict = field(default_factory=dict)

    # Optional contrast (non-refusal) concept vectors for detector work
    contrast_vectors: Optional[dict[int, dict[str, torch.Tensor]]] = None

    def save(self, path: str) -> None:
        """Save to disk. All tensors moved to CPU."""
        save_dict = {
            "concept_vectors": {
                l: {c: v.cpu() for c, v in concepts.items()}
                for l, concepts in self.concept_vectors.items()
            },
            "layer_signal": {int(k): float(v) for k, v in self.layer_signal.items()},
            "basis": {l: v.cpu() for l, v in self.basis.items()},
            "metadata": self.metadata,
            "contrast_vectors": (
                {
                    l: {c: v.cpu() for c, v in concepts.items()}
                    for l, concepts in self.contrast_vectors.items()
                }
                if self.contrast_vectors is not None
                else None
            ),
        }
        torch.save(save_dict, path)
        logger.info(f"Saved J-lens vectors to {path}")

    @classmethod
    def load(cls, path: str) -> "JLensVectors":
        """Load from disk. Uses weights_only=False since payload contains dict metadata."""
        # weights_only=False because metadata may contain non-tensor structures
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            concept_vectors=data["concept_vectors"],
            layer_signal={int(k): float(v) for k, v in data["layer_signal"].items()},
            basis=data["basis"],
            metadata=data.get("metadata", {}),
            contrast_vectors=data.get("contrast_vectors"),
        )


class JLensExtractor:
    """Runs grad-enabled forward passes, captures intermediate hidden states,
    computes VJPs for each concept token, and accumulates per-layer per-concept
    means via Welford."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: JLensConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # Populated during a single forward pass, cleared between iterations
        self._captured: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        # (layer_idx, concept_str) -> accumulator
        self._welford: dict[tuple[int, str], WelfordMeanAccumulator] = {}

    def _make_hook(self, layer_idx: int):
        extractor = self  # closure over self so we always see the current _captured dict

        def hook(module, inputs, output):
            # Output may be tuple (hidden_states, ...) or a bare Tensor
            h = output[0] if isinstance(output, tuple) else output
            # Do NOT detach — we need the autograd graph
            extractor._captured[layer_idx] = h

        return hook

    def _register_hooks(self, layer_indices: list[int]) -> None:
        layers = get_transformer_layers(self.model)
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _concept_token_ids(self, concepts: list[str]) -> dict[str, int]:
        """Map each concept string to its first-token id, dropping unknowns."""
        out: dict[str, int] = {}
        for c in concepts:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if not ids:
                logger.warning(f"J-lens: concept {c!r} encodes to empty; skipping")
                continue
            out[c] = ids[0]
        return out

    def _format_batch(self, prompts: list[str]) -> list[str]:
        # apply_chat_template exists on all tokenizers but raises if no template is set.
        # Base LMs (GPT-2 etc.) have no chat template — fall back to raw prompts.
        if getattr(self.tokenizer, "chat_template", None):
            return [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        return list(prompts)

    def extract(
        self,
        prompts: list[str],
        layer_indices: Optional[list[int]] = None,
        concepts: Optional[list[str]] = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Compute per-layer per-concept J-lens vectors averaged over `prompts`.

        Returns dict[layer_idx][concept_str] -> [hidden_dim] float32 CPU tensor.
        """
        self.model.eval()
        # We use torch.autograd.grad (not .backward), so no grad accumulates in
        # param.grad. But params MUST have requires_grad=True so intermediate
        # activations remain in the autograd graph (token-id inputs are integer
        # and non-differentiable). Snapshot + restore the flags either way.
        _param_grad_flags = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in _param_grad_flags:
            p.requires_grad_(True)

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                logger.warning(f"J-lens: gradient_checkpointing_enable failed: {e}")

        num_layers = len(get_transformer_layers(self.model))
        if layer_indices is None:
            layer_indices = self.config.layer_indices
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        concepts = concepts or self.config.concepts
        concept_ids = self._concept_token_ids(concepts)
        if not concept_ids:
            raise ValueError("J-lens: no valid concept tokens after tokenization")

        self._welford = {}
        self._register_hooks(layer_indices)
        num_prompts_seen = 0

        try:
            for i in tqdm(
                range(0, len(prompts), self.config.batch_size),
                desc="J-lens VJPs",
            ):
                batch = prompts[i : i + self.config.batch_size]
                formatted = self._format_batch(batch)

                orig_pad = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len,
                ).to(self.config.device)
                self.tokenizer.padding_side = orig_pad

                self._captured.clear()
                with torch.enable_grad():
                    outputs = self.model(**inputs, use_cache=False)
                    logits = outputs.logits  # [batch, seq, vocab]
                    last_logits = logits[:, -1, :]  # [batch, vocab]

                    captured_layers = sorted(self._captured.keys())
                    captured_tensors = [self._captured[l] for l in captured_layers]
                    if not captured_tensors:
                        logger.warning("J-lens: no hidden states captured this batch; skipping")
                        continue

                    concept_items = list(concept_ids.items())
                    for j, (concept, token_id) in enumerate(concept_items):
                        # Sum across batch to get a scalar for autograd
                        L_c = last_logits[:, token_id].sum()
                        is_last = j == len(concept_items) - 1
                        grads = torch.autograd.grad(
                            L_c,
                            captured_tensors,
                            retain_graph=not is_last,
                            allow_unused=True,
                        )
                        for layer_idx, grad in zip(captured_layers, grads):
                            if grad is None:
                                continue
                            # Last-position gradient per batch element
                            last_grad = grad[:, -1, :].detach().to(
                                dtype=torch.float32, device="cpu"
                            )
                            key = (layer_idx, concept)
                            if key not in self._welford:
                                self._welford[key] = WelfordMeanAccumulator(
                                    hidden_dim=last_grad.shape[-1]
                                )
                            self._welford[key].update(last_grad)

                num_prompts_seen += len(batch)
                # Explicit cleanup to keep peak memory sane on 12B models
                del outputs, logits, last_logits, captured_tensors
                self._captured.clear()
                if self.config.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        finally:
            self._remove_hooks()
            if self.config.gradient_checkpointing and hasattr(
                self.model, "gradient_checkpointing_disable"
            ):
                try:
                    self.model.gradient_checkpointing_disable()
                except Exception:
                    pass
            # Restore original requires_grad flags so downstream inference is unaffected
            for p, flag in _param_grad_flags:
                p.requires_grad_(flag)
                if p.grad is not None:
                    p.grad = None

        # Assemble result
        result: dict[int, dict[str, torch.Tensor]] = {}
        for (layer_idx, concept), acc in self._welford.items():
            result.setdefault(layer_idx, {})[concept] = acc.get_mean()
        logger.info(
            f"J-lens: computed vectors for {len(result)} layers x "
            f"{len(concept_ids)} concepts over {num_prompts_seen} prompts"
        )
        return result


def compute_layer_signal(
    concept_vectors: dict[int, dict[str, torch.Tensor]],
) -> dict[int, float]:
    """Per-layer refusal-signal strength = mean L2 norm of concept vectors, normalized so max=1.0."""
    raw: dict[int, float] = {}
    for layer_idx, concepts in concept_vectors.items():
        if not concepts:
            raw[layer_idx] = 0.0
            continue
        norms = [float(v.norm().item()) for v in concepts.values()]
        raw[layer_idx] = sum(norms) / len(norms)

    max_val = max(raw.values()) if raw else 0.0
    if max_val <= 0.0:
        return {k: 0.0 for k in raw}
    return {k: v / max_val for k, v in raw.items()}


def build_jlens_basis(
    concept_vectors: dict[int, dict[str, torch.Tensor]],
    rank: Optional[int] = None,
    rank_tol: float = 1e-6,
) -> dict[int, torch.Tensor]:
    """Per-layer QR orthonormalization of concept vectors. Returns [hidden_dim, k] per layer."""
    basis: dict[int, torch.Tensor] = {}
    for layer_idx, concepts in concept_vectors.items():
        vectors = [v for v in concepts.values() if v.norm() > 0]
        if not vectors:
            continue
        # [hidden_dim, num_concepts]
        M = torch.stack(vectors, dim=1).to(dtype=torch.float32)
        Q, R = torch.linalg.qr(M, mode="reduced")
        # Drop columns whose R diagonal is near zero (linearly dependent inputs)
        diag = R.diagonal().abs()
        keep = diag > rank_tol
        Q = Q[:, keep]
        if rank is not None and Q.shape[1] > rank:
            Q = Q[:, :rank]
        if Q.shape[1] > 0:
            basis[layer_idx] = Q
    return basis


def compute_jlens_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    config: JLensConfig,
    include_contrast: bool = False,
) -> JLensVectors:
    """High-level entry point: extract concept vectors, derive signal + basis, package as JLensVectors."""
    if config.num_prompts is not None and len(prompts) > config.num_prompts:
        prompts = prompts[: config.num_prompts]

    extractor = JLensExtractor(model, tokenizer, config)
    concept_vectors = extractor.extract(prompts, concepts=config.concepts)
    contrast_vectors = None
    if include_contrast and config.contrast_concepts:
        contrast_vectors = extractor.extract(prompts, concepts=config.contrast_concepts)

    layer_signal = compute_layer_signal(concept_vectors)
    basis = build_jlens_basis(concept_vectors, rank=config.basis_rank)

    metadata = {
        "concepts": list(config.concepts),
        "contrast_concepts": list(config.contrast_concepts) if include_contrast else [],
        "num_prompts": len(prompts),
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "basis_rank": config.basis_rank,
    }
    return JLensVectors(
        concept_vectors=concept_vectors,
        layer_signal=layer_signal,
        basis=basis,
        metadata=metadata,
        contrast_vectors=contrast_vectors,
    )


def generate_layer_target_map(
    jlens: JLensVectors,
    num_layers: int,
    exclude_threshold: float = 0.2,
    min_multiplier: float = 0.1,
    aggressive_threshold: float = 0.8,
    model_path: str = "",
) -> dict:
    """Convert per-layer J-lens signal into a layer_target_map JSON structure
    compatible with abliterate.load_layer_target_map.
    """
    signal = jlens.layer_signal
    per_layer_multipliers: dict[int, float] = {}
    excluded: list[int] = []
    aggressive: list[int] = []

    for i in range(num_layers):
        s = float(signal.get(i, 0.0))
        if s < exclude_threshold:
            excluded.append(i)
        else:
            per_layer_multipliers[i] = float(max(s, min_multiplier))
            if s >= aggressive_threshold:
                aggressive.append(i)

    # Derive recommended Gaussian params from signal distribution as a fallback
    if signal and num_layers > 1:
        total = sum(signal.values())
        if total > 0:
            center_ratio = sum((i / (num_layers - 1)) * s for i, s in signal.items()) / total
            var = (
                sum(((i / (num_layers - 1)) - center_ratio) ** 2 * s for i, s in signal.items())
                / total
            )
            sigma_ratio = math.sqrt(max(var, 1e-6))
        else:
            center_ratio, sigma_ratio = 0.6, 0.2
    else:
        center_ratio, sigma_ratio = 0.6, 0.2

    layer_stats = {}
    for i in range(num_layers):
        concepts = jlens.concept_vectors.get(i, {})
        layer_stats[str(i)] = {
            "jlens_signal": float(signal.get(i, 0.0)),
            "concept_norms": {c: float(v.norm().item()) for c, v in concepts.items()},
        }

    return {
        "version": "1.0-jlens",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "layer_multipliers": {str(k): float(v) for k, v in per_layer_multipliers.items()},
        "excluded_layers": excluded,
        "target_layer_indices": sorted(per_layer_multipliers.keys()),
        "recommended_center_ratio": float(center_ratio),
        "recommended_sigma_ratio": float(sigma_ratio),
        "aggressive_layers": aggressive,
        "protected_layers": [],
        "layer_stats": layer_stats,
        "analysis_params": {
            "source": "jlens",
            "concepts": jlens.metadata.get("concepts", []),
            "num_prompts": jlens.metadata.get("num_prompts", 0),
            "exclude_threshold": exclude_threshold,
            "min_multiplier": min_multiplier,
            "aggressive_threshold": aggressive_threshold,
            "model_path": model_path,
        },
    }


def restrict_direction_to_jlens_subspace(
    direction: torch.Tensor,
    basis: torch.Tensor,
    min_projection_ratio: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Project `direction` onto the column-space of `basis` (assumed orthonormal).

    Returns (restricted_direction, retained_norm_ratio). If the retained ratio
    falls below `min_projection_ratio`, returns the ORIGINAL direction unchanged
    with the true ratio — the caller uses this as a fallback signal.
    """
    if basis is None or basis.numel() == 0:
        return direction, 1.0
    d = direction.to(dtype=torch.float32, device=basis.device)
    orig_norm = d.norm()
    if orig_norm.item() == 0.0:
        return direction, 0.0
    # d_restricted = B @ (B^T @ d)
    coeffs = basis.T @ d
    d_restricted = basis @ coeffs
    restricted_norm = d_restricted.norm()
    ratio = float((restricted_norm / orig_norm).item())
    if ratio < min_projection_ratio:
        logger.warning(
            f"J-lens restriction retained ratio={ratio:.3f} < min={min_projection_ratio}; "
            f"falling back to unrestricted direction"
        )
        return direction, ratio
    return d_restricted.to(dtype=direction.dtype, device=direction.device), ratio


def save_target_map_json(target_map: dict, path: str) -> None:
    """Write target map to JSON with pretty formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(target_map, f, indent=2, sort_keys=False)
    logger.info(f"Wrote layer target map to {path}")
