"""Workspace-based refusal detection.

Reads refusal concepts DIRECTLY from intermediate hidden states via projection
onto J-lens concept vectors. Complements LogLikelihoodRefusalDetector:

- Log-likelihood detector: measures *output* refusal disposition (how likely
  the model is to *say* "I cannot").
- Workspace detector: measures *internal* refusal deliberation (how strongly
  the model is holding refusal concepts in its workspace).

The interesting diagnostic case is when they DISAGREE — a model whose output
looks compliant but whose workspace still lights up on refusal concepts has
had its behavior suppressed at the surface without removing the underlying
computation (Hydra effect / self-repair).

Public API mirrors LogLikelihoodRefusalDetector so it can be a drop-in
replacement in RefusalScanner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.abliterate import get_transformer_layers
from src.jlens import JLensVectors

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceRefusalDetectorConfig:
    """Configuration for the workspace-based refusal detector."""

    jlens_vectors_path: str = ""              # Required — path to jlens_vectors.pt
    active_layers: Optional[list[int]] = None # If None: layers with signal > active_signal_threshold
    active_signal_threshold: float = 0.5      # Fraction of max signal to be "active"
    projection_threshold: float = 0.3         # cosine-projection threshold to flag refusal
    aggregation: str = "max"                  # "max" | "mean" across active layers
    concept_aggregation: str = "max"          # "max" | "mean" across concepts within a layer


class WorkspaceRefusalDetector:
    """Detects refusal by projecting mid-computation hidden states onto J-lens
    refusal-concept vectors. Requires a precomputed JLensVectors on disk."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: WorkspaceRefusalDetectorConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if not config.jlens_vectors_path:
            raise ValueError("WorkspaceRefusalDetectorConfig requires jlens_vectors_path")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load J-lens vectors
        self.jlens: JLensVectors = JLensVectors.load(config.jlens_vectors_path)

        # Determine which layers are "workspace-active"
        if config.active_layers is not None:
            self.active_layers = sorted(config.active_layers)
        else:
            max_signal = max(self.jlens.layer_signal.values(), default=1.0) or 1.0
            self.active_layers = sorted(
                l for l, s in self.jlens.layer_signal.items()
                if s / max_signal >= config.active_signal_threshold
                and l in self.jlens.concept_vectors
            )
        if not self.active_layers:
            raise ValueError(
                "No workspace-active layers found; check active_signal_threshold or J-lens data"
            )
        logger.info(
            f"Workspace detector: using {len(self.active_layers)} active layers "
            f"({self.active_layers})"
        )

        # Pre-normalize concept vectors per active layer and stack for cheap dot-product
        # {layer: Tensor[num_concepts, hidden_dim]}
        self._concept_matrix: dict[int, torch.Tensor] = {}
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for layer in self.active_layers:
            concepts = self.jlens.concept_vectors.get(layer, {})
            if not concepts:
                continue
            vecs = []
            for v in concepts.values():
                n = v.norm()
                if n.item() > 0:
                    vecs.append(v / n)
            if not vecs:
                continue
            M = torch.stack(vecs, dim=0).to(device=device, dtype=dtype)  # [K, D]
            self._concept_matrix[layer] = M

        # Populated by forward hooks
        self._captured: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    # --- Hook management -------------------------------------------------

    def _make_hook(self, layer_idx: int):
        detector = self

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Last-position hidden state per batch element, detached (no grad needed)
            detector._captured[layer_idx] = h[:, -1, :].detach()

        return hook

    def _register_hooks(self) -> None:
        layers = get_transformer_layers(self.model)
        for idx in self.active_layers:
            if 0 <= idx < len(layers):
                self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # --- Public API (mirrors LogLikelihoodRefusalDetector) ----------------

    def format_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _score_batch(self, prompts: list[str]) -> torch.Tensor:
        """Compute workspace refusal scores for a batch of prompts (higher = more refusal deliberation)."""
        formatted = [self.format_prompt(p) for p in prompts]

        original_padding = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                formatted, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)
        finally:
            self.tokenizer.padding_side = original_padding

        self._captured.clear()
        self._register_hooks()
        try:
            with torch.no_grad():
                _ = self.model(**inputs, use_cache=False)
        finally:
            self._remove_hooks()

        batch_size = inputs.input_ids.shape[0]
        # per_layer_scores[layer] -> [batch]
        layer_scores: list[torch.Tensor] = []
        for layer in self.active_layers:
            if layer not in self._captured or layer not in self._concept_matrix:
                continue
            h = self._captured[layer]                 # [B, D]
            h_norm = F.normalize(h.to(torch.float32), dim=-1)
            M = self._concept_matrix[layer].to(torch.float32)  # [K, D] already normalized
            projs = h_norm @ M.T                       # [B, K] cosine similarities
            # Concept aggregation
            if self.config.concept_aggregation == "mean":
                per_layer = projs.mean(dim=-1)
            else:
                per_layer = projs.abs().max(dim=-1).values
            layer_scores.append(per_layer.cpu())

        if not layer_scores:
            return torch.zeros(batch_size)

        stacked = torch.stack(layer_scores, dim=0)     # [num_active_layers, B]
        if self.config.aggregation == "mean":
            return stacked.mean(dim=0)
        return stacked.max(dim=0).values

    def detect_refusal_batch(self, prompts: list[str]) -> list[bool]:
        scores = self._score_batch(prompts)
        return (scores > self.config.projection_threshold).tolist()

    def detect_refusal(self, prompt: str) -> bool:
        return self.detect_refusal_batch([prompt])[0]

    def detect_refusal_with_scores(
        self, prompts: list[str]
    ) -> list[tuple[bool, float]]:
        scores = self._score_batch(prompts)
        return [
            (float(s) > self.config.projection_threshold, float(s))
            for s in scores.tolist()
        ]

    def detect_refusal_with_score(self, prompt: str) -> tuple[bool, float]:
        return self.detect_refusal_with_scores([prompt])[0]
