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
class MinedConcept:
    """Result of contrastive concept mining."""

    concept: str            # Decoded token string (also used as dict key downstream)
    token_id: int           # Vocabulary id (canonical — bypasses re-tokenization)
    specificity: float      # mean P(t | harmful) - mean P(t | harmless)
    p_harmful: float
    p_harmless: float


def _apply_chat_template_no_think(tokenizer, messages: list[dict]) -> str:
    """Apply chat template with add_generation_prompt=True and thinking disabled.

    Reasoning-mode models (Qwen3/Qwen3.5, DeepSeek-R1, etc.) inject a `<think>`
    prefix by default, which pushes the model into a reasoning preamble instead
    of directly emitting the refusal (or acceptance). Passing enable_thinking=False
    forces the response to start at position 0.

    Falls back to the plain call for tokenizers that don't accept the kwarg.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _apply_chat_template_or_raw(tokenizer, prompts: list[str]) -> list[str]:
    if getattr(tokenizer, "chat_template", None):
        return [
            _apply_chat_template_no_think(tokenizer, [{"role": "user", "content": p}])
            for p in prompts
        ]
    return list(prompts)


def _mean_next_token_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
    device: str,
) -> torch.Tensor:
    """Average softmax(logits[:, -1, :]) across `prompts`. Returns [vocab] on CPU float32."""
    acc: Optional[torch.Tensor] = None
    total = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Mining next-token probs"):
        batch = prompts[i : i + batch_size]
        formatted = _apply_chat_template_or_raw(tokenizer, batch)
        orig_pad = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        tokenizer.padding_side = orig_pad
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
            probs = outputs.logits[:, -1, :].float().softmax(dim=-1)  # [batch, vocab]
        summed = probs.sum(dim=0).detach().to("cpu")
        acc = summed if acc is None else acc + summed
        total += probs.shape[0]
        del outputs, probs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if acc is None or total == 0:
        raise ValueError("No prompts consumed by mining pass")
    return acc / total


def _mine_next_token_stats_multi(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
    device: str,
    per_prompt_topk: int = 5,
    num_positions: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Multi-position streaming stats via teacher-forced argmax walks.

    For each prompt, does `num_positions` forward passes: at each step, records the
    softmax over the last position, then teacher-forces the argmax token as the
    next input. This walks down the greedy refusal continuation
    ("I" -> "cannot" -> "help" -> "with"...) and lets multiple concepts surface
    even when position 0 is dominated by a single token.

    Aggregation:
      - prob_sum[t] = sum over prompts of (max over positions of P(t | prefix))
      - coverage[t] = count of prompts where t appears in per-prompt top-K
                     at ANY of the visited positions
    """
    prob_sum: Optional[torch.Tensor] = None
    coverage: Optional[torch.Tensor] = None
    total = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Mining next-token stats"):
        batch = prompts[i : i + batch_size]
        formatted = _apply_chat_template_or_raw(tokenizer, batch)
        orig_pad = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        tokenizer.padding_side = orig_pad
        current_ids = inputs.input_ids
        current_mask = inputs.attention_mask
        b = current_ids.shape[0]
        vocab_size = getattr(model.config, "vocab_size", None)

        per_prompt_max: Optional[torch.Tensor] = None
        per_prompt_seen: Optional[torch.Tensor] = None

        for _ in range(num_positions):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids, attention_mask=current_mask, use_cache=False,
                )
                probs = outputs.logits[:, -1, :].float().softmax(dim=-1)  # [b, V]
            if per_prompt_max is None:
                per_prompt_max = probs.clone()
            else:
                per_prompt_max = torch.maximum(per_prompt_max, probs)
            _, top_idx = probs.topk(per_prompt_topk, dim=-1)
            in_top_k = torch.zeros_like(probs)
            in_top_k.scatter_(1, top_idx, 1.0)
            if per_prompt_seen is None:
                per_prompt_seen = in_top_k.clone()
            else:
                per_prompt_seen = torch.maximum(per_prompt_seen, in_top_k)
            # Teacher-force top-1 for the next step
            next_tokens = probs.argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
            current_mask = torch.cat([current_mask, torch.ones_like(next_tokens)], dim=1)
            del outputs, probs, in_top_k

        batch_prob = per_prompt_max.sum(dim=0).detach().to("cpu")
        batch_cov = per_prompt_seen.sum(dim=0).detach().to("cpu")
        prob_sum = batch_prob if prob_sum is None else prob_sum + batch_prob
        coverage = batch_cov if coverage is None else coverage + batch_cov
        total += b
        del per_prompt_max, per_prompt_seen
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if prob_sum is None or total == 0:
        raise ValueError("No prompts consumed by mining pass")
    return prob_sum, coverage, total


def _mine_next_token_stats(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
    device: str,
    per_prompt_topk: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Single-pass streaming stats for the last-position softmax across `prompts`.

    Returns (prob_sum, coverage, n_prompts) where
      - prob_sum[t] = sum over prompts of P(t | prompt)
      - coverage[t] = count of prompts where t is in the per-prompt top-K
    """
    prob_sum: Optional[torch.Tensor] = None
    coverage: Optional[torch.Tensor] = None
    total = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Mining next-token stats"):
        batch = prompts[i : i + batch_size]
        formatted = _apply_chat_template_or_raw(tokenizer, batch)
        orig_pad = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        tokenizer.padding_side = orig_pad
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
            probs = outputs.logits[:, -1, :].float().softmax(dim=-1)  # [batch, vocab]
        # Coverage: mark per-prompt top-K positions
        _, top_idx = probs.topk(per_prompt_topk, dim=-1)
        in_top_k = torch.zeros_like(probs)
        in_top_k.scatter_(1, top_idx, 1.0)

        batch_prob = probs.sum(dim=0).detach().to("cpu")
        batch_cov = in_top_k.sum(dim=0).detach().to("cpu")
        prob_sum = batch_prob if prob_sum is None else prob_sum + batch_prob
        coverage = batch_cov if coverage is None else coverage + batch_cov
        total += probs.shape[0]
        del outputs, probs, in_top_k
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if prob_sum is None or total == 0:
        raise ValueError("No prompts consumed by mining pass")
    return prob_sum, coverage, total


def _filter_and_build(
    tokenizer,
    scores: torch.Tensor,
    top_k: int,
    min_score: float,
    exclude_tokens: Optional[set[str]],
    score_name: str,
    extras: dict[int, dict],
) -> list[MinedConcept]:
    """Shared post-processing: over-fetch, decode, filter, build MinedConcept list."""
    fetch_k = min(max(top_k * 6, top_k + 20), scores.shape[0])
    top = torch.topk(scores, k=fetch_k)
    exclude_tokens = set(exclude_tokens or set())
    results: list[MinedConcept] = []
    seen_norm: set[str] = set()
    for score, tid in zip(top.values.tolist(), top.indices.tolist()):
        if score < min_score:
            break
        try:
            concept_str = tokenizer.decode([tid])
        except Exception:
            continue
        norm = concept_str.strip()
        if not norm or norm in exclude_tokens or norm in seen_norm:
            continue
        if len(norm) == 1 and not norm.isalnum():
            continue
        seen_norm.add(norm)
        extra = extras.get(int(tid), {})
        results.append(MinedConcept(
            concept=concept_str,
            token_id=int(tid),
            specificity=float(score),
            p_harmful=float(extra.get("p_harmful", 0.0)),
            p_harmless=float(extra.get("p_harmless", 0.0)),
        ))
        if len(results) >= top_k:
            break
    return results


def mine_refusal_concepts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_prompts: list[str],
    harmless_prompts: Optional[list[str]] = None,
    top_k: int = 8,
    min_specificity: float = 0.005,
    batch_size: int = 4,
    max_seq_len: int = 128,
    device: str = "cuda",
    exclude_tokens: Optional[set[str]] = None,
    mode: str = "consensus",
    per_prompt_topk: int = 5,
    min_coverage: float = 0.2,
    num_positions: int = 4,
) -> list[MinedConcept]:
    """Discover model-specific refusal-token concepts.

    Two strategies:

    - "consensus" (default): find tokens that MANY harmful prompts converge on,
      independent of what harmless prompts do. Score = coverage * mean_prob,
      where coverage is the fraction of harmful prompts where the token is in
      the per-prompt top-K. Rationale: refusals are the mode of the harmful-
      prompt response distribution; helpful responses to diverse harmless
      prompts diverge, so subtracting them is uninformative.

    - "contrast": specificity = mean P(t | harmful) - mean P(t | harmless).
      Discriminates against dual-use tokens (" I", " can") that appear in
      both response types. Requires harmless_prompts.

    Args:
        mode: "consensus" (default) or "contrast"
        top_k: max concepts to return
        min_specificity: skip tokens whose final score is below this
        per_prompt_topk: (consensus) top-K per prompt for coverage counting
        min_coverage: (consensus) minimum fraction of prompts where the token
                      must be in per-prompt top-K (default 0.2 = 20%)
        exclude_tokens: iterable of visible strings (after strip) to filter out
    """
    model.eval()
    if mode == "consensus":
        logger.info(
            f"Concept mining (consensus, multi-position): {len(harmful_prompts)} harmful prompts, "
            f"positions={num_positions}, per-position top-K={per_prompt_topk}, "
            f"min_coverage={min_coverage}"
        )
        prob_sum, coverage, n = _mine_next_token_stats_multi(
            model, tokenizer, harmful_prompts, batch_size, max_seq_len, device,
            per_prompt_topk=per_prompt_topk, num_positions=num_positions,
        )
        mean_prob = prob_sum / n              # [vocab] — mean of per-prompt max prob across visited positions
        cov_frac = coverage / n               # [vocab] — fraction of prompts where t is in top-K at some position
        # Score: multiplicative — both presence AND consistent presence matter
        score = cov_frac * mean_prob
        # Zero out tokens below coverage threshold
        score = torch.where(cov_frac >= min_coverage, score, torch.zeros_like(score))
        # Build the "extras" map (p_harmful used as mean_prob; p_harmless left 0)
        # Only compute for the top candidates to save time
        fetch_k = min(max(top_k * 6, top_k + 20), score.shape[0])
        top_ids = torch.topk(score, k=fetch_k).indices.tolist()
        extras = {tid: {"p_harmful": float(mean_prob[tid].item()), "p_harmless": float(cov_frac[tid].item())}
                  for tid in top_ids}
        results = _filter_and_build(
            tokenizer, score, top_k, min_specificity, exclude_tokens,
            score_name="score", extras=extras,
        )
        # For consensus, `specificity` on MinedConcept is the score (cov * mean_prob);
        # p_harmful holds mean_prob, p_harmless holds cov_frac. Display code labels these.

    elif mode == "contrast":
        if harmless_prompts is None:
            raise ValueError("contrast mode requires harmless_prompts")
        logger.info(
            f"Concept mining (contrast): {len(harmful_prompts)} harmful, "
            f"{len(harmless_prompts)} harmless prompts"
        )
        harm_sum, _, n_harm = _mine_next_token_stats(
            model, tokenizer, harmful_prompts, batch_size, max_seq_len, device,
        )
        harmless_sum, _, n_harmless = _mine_next_token_stats(
            model, tokenizer, harmless_prompts, batch_size, max_seq_len, device,
        )
        p_harm = harm_sum / n_harm
        p_harmless = harmless_sum / n_harmless
        specificity = p_harm - p_harmless
        extras = {}
        # For contrast, populate p_harmful / p_harmless for each candidate
        fetch_k = min(max(top_k * 6, top_k + 20), specificity.shape[0])
        top_ids = torch.topk(specificity, k=fetch_k).indices.tolist()
        extras = {tid: {"p_harmful": float(p_harm[tid].item()),
                        "p_harmless": float(p_harmless[tid].item())}
                  for tid in top_ids}
        results = _filter_and_build(
            tokenizer, specificity, top_k, min_specificity, exclude_tokens,
            score_name="specificity", extras=extras,
        )
    else:
        raise ValueError(f"Unknown mining mode: {mode!r} (expected 'consensus' or 'contrast')")

    if not results:
        logger.warning(
            "Concept mining produced no candidates above min_score=%.4f (mode=%s). "
            "This can happen if the model no longer refuses (already abliterated) "
            "or if its refusal openers are highly varied.",
            min_specificity, mode,
        )
    else:
        logger.info(
            "Mined %d refusal concepts (mode=%s): %s",
            len(results), mode,
            ", ".join(f"{r.concept!r} ({r.specificity:.4f})" for r in results),
        )
    return results


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
    # Optional pre-resolved concept -> token_id mapping (from mining).
    # When set, the extractor uses these ids directly, bypassing re-tokenization
    # of the concept string (which can be unstable for BPE tokenizers).
    concept_token_ids: Optional[dict[str, int]] = None


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
        """Map each concept string to its first-token id, dropping unknowns.

        If self.config.concept_token_ids is set, prefer those direct id mappings
        over re-tokenizing the string (avoids BPE round-trip drift for tokens
        discovered via contrastive mining).
        """
        preset = self.config.concept_token_ids or {}
        out: dict[str, int] = {}
        for c in concepts:
            if c in preset:
                out[c] = int(preset[c])
                continue
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if not ids:
                logger.warning(f"J-lens: concept {c!r} encodes to empty; skipping")
                continue
            out[c] = ids[0]
        return out

    def _format_batch(self, prompts: list[str]) -> list[str]:
        # apply_chat_template exists on all tokenizers but raises if no template is set.
        # Base LMs (GPT-2 etc.) have no chat template — fall back to raw prompts.
        # Reasoning-mode models get thinking disabled (see _apply_chat_template_no_think).
        if getattr(self.tokenizer, "chat_template", None):
            return [
                _apply_chat_template_no_think(
                    self.tokenizer, [{"role": "user", "content": p}]
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


def build_jspace_config_dict(
    mode: str,
    model_path: str,
    output_path: str,
    device: str,
    dtype: str,
    jlens_params: dict,
    concepts_params: dict,
    target_map_params: Optional[dict] = None,
    abliteration_params: Optional[dict] = None,
    iterative_params: Optional[dict] = None,
) -> dict:
    """Assemble a J-space-specific abliteration_config.json payload.

    Structure is mode-aware: each section is present only when it applies.

    Args:
        mode: "map", "restrict", or "iterative"
        jlens_params: J-lens extraction settings (num_prompts, batch_size, basis_rank, ...)
        concepts_params: source + mining settings + list of mined concepts (if any)
        target_map_params: exclude_threshold, min_multiplier, layer counts (map mode)
        abliteration_params: direction_multiplier, winsorization, etc. (restrict/iterative)
        iterative_params: loop settings + convergence result (iterative)
    """
    payload = {
        "abliteration_type": "j-space",
        "j_space_mode": mode,
        "model_path": model_path,
        "output_path": str(output_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device,
        "dtype": dtype,
        "j_lens": jlens_params,
        "concepts": concepts_params,
    }
    if target_map_params is not None:
        payload["target_map"] = target_map_params
    if abliteration_params is not None:
        payload["abliteration"] = abliteration_params
    if iterative_params is not None:
        payload["iterative"] = iterative_params
    return payload


def save_jspace_config_json(payload: dict, output_path: str) -> None:
    """Write J-space abliteration config to <output_path>/abliteration_config.json."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "abliteration_config.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
    logger.info(f"Wrote J-space config to {dest}")
