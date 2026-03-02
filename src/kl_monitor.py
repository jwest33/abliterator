#!/usr/bin/env python3
"""
KL Divergence Monitor & Auto-Tune Multiplier

Measures distribution drift between original and abliterated model outputs
on reference prompts (typically preservation/harmless prompts). Uses top-k
approximation for efficient KL computation without full vocab softmax.

Provides:
- KL monitoring: post-abliteration quality signal reporting distribution drift
- Auto-tune: binary search over direction_multiplier to find highest value
  that keeps KL below a user-specified threshold
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration & Result Dataclasses
# ==============================================================================


@dataclass
class KLMonitorConfig:
    """Configuration for KL divergence monitoring."""

    num_reference_prompts: int = 50
    top_k: int = 200  # Top-k tokens for KL approximation
    batch_size: int = 4

    # Auto-tune binary search bounds
    search_min: float = 0.1
    search_max: float = 2.0
    search_tolerance: float = 0.01
    max_search_iterations: int = 15
    kl_threshold: float = 0.5  # Max mean KL (nats)


@dataclass
class KLResult:
    """Result of KL divergence computation."""

    mean_kl: float
    median_kl: float
    max_kl: float
    std_kl: float
    per_prompt_kl: list[float]
    multiplier: float  # The direction_multiplier used

    def to_dict(self) -> dict:
        return {
            "mean_kl": self.mean_kl,
            "median_kl": self.median_kl,
            "max_kl": self.max_kl,
            "std_kl": self.std_kl,
            "per_prompt_kl": self.per_prompt_kl,
            "multiplier": self.multiplier,
        }


@dataclass
class AutoTuneResult:
    """Result of auto-tune binary search."""

    best_multiplier: float
    best_kl: float
    search_history: list[dict]  # [{multiplier, mean_kl}, ...]
    converged: bool
    num_iterations: int
    kl_threshold: float

    def to_dict(self) -> dict:
        return {
            "best_multiplier": self.best_multiplier,
            "best_kl": self.best_kl,
            "search_history": self.search_history,
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "kl_threshold": self.kl_threshold,
        }


# ==============================================================================
# Weight Snapshot (for iterative rollback during auto-tune)
# ==============================================================================


class WeightSnapshot:
    """
    Saves and restores linear layer weights for iterative rollback.

    Stores weight clones on CPU to avoid doubling GPU memory.
    For 7B models: ~13GB CPU RAM. For 70B+: ~140GB.
    """

    def __init__(self, model: AutoModelForCausalLM, linear_layer_names: list[str]):
        self.snapshots: dict[str, torch.Tensor] = {}
        self._save(model, linear_layer_names)

    def _save(self, model: AutoModelForCausalLM, linear_layer_names: list[str]):
        """Save current weights of all linear layers to CPU."""
        name_to_module = dict(model.named_modules())
        for name in linear_layer_names:
            module = name_to_module.get(name)
            if module is not None and hasattr(module, "weight"):
                self.snapshots[name] = module.weight.data.clone().cpu()
        logger.info(f"Snapshot saved: {len(self.snapshots)} linear layers on CPU")

    def restore(self, model: AutoModelForCausalLM):
        """Restore saved weights back to the model (in-place)."""
        name_to_module = dict(model.named_modules())
        restored = 0
        for name, saved_weight in self.snapshots.items():
            module = name_to_module.get(name)
            if module is not None and hasattr(module, "weight"):
                module.weight.data.copy_(saved_weight.to(module.weight.device))
                restored += 1
        logger.info(f"Restored {restored}/{len(self.snapshots)} linear layer weights")

    def free(self):
        """Release CPU memory held by snapshots."""
        num = len(self.snapshots)
        self.snapshots.clear()
        logger.info(f"Freed weight snapshot ({num} layers)")


# ==============================================================================
# KL Divergence Monitor
# ==============================================================================


class KLDivergenceMonitor:
    """
    Monitors KL divergence between original and abliterated model distributions.

    Usage:
        1. Create monitor with original model
        2. Call cache_reference_logits() BEFORE abliteration
        3. After abliteration, call compute_kl_divergence() to measure drift

    Uses top-k approximation: only computes KL over the top-k tokens from the
    original distribution, capturing >99% of probability mass while avoiding
    full 128k+ vocab softmax storage per position.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: KLMonitorConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Cached reference logits: list of (top_k_log_probs, top_k_indices) per prompt
        # Each is (num_positions, top_k) stored on CPU
        self.cached_references: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Ensure padding is configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _forward_batch(self, prompts: list[str]) -> list[torch.Tensor]:
        """
        Run forward pass on a batch of prompts, returning per-prompt logits.

        Returns list of logit tensors, each (seq_len, vocab_size), trimmed
        to exclude padding positions.
        """
        formatted = [self._format_prompt(p) for p in prompts]

        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                formatted, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (batch, seq_len, vocab)

            # Extract per-prompt logits, excluding left-padding
            attention_mask = inputs.attention_mask  # (batch, seq_len)
            results = []
            for i in range(logits.size(0)):
                # Find first non-padding position
                mask = attention_mask[i]
                non_pad = mask.nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    start = non_pad[0].item()
                    results.append(logits[i, start:].cpu())
                else:
                    results.append(logits[i].cpu())

            # Free GPU tensor before returning
            del logits, outputs, inputs
            return results
        finally:
            self.tokenizer.padding_side = original_padding_side

    def cache_reference_logits(self, prompts: list[str]):
        """
        Cache top-k log probabilities from the original (pre-abliteration) model.

        Must be called BEFORE abliteration since the model is modified in-place.

        Args:
            prompts: Reference prompts (typically preservation or harmless prompts)
        """
        self.cached_references = []
        top_k = self.config.top_k
        batch_size = self.config.batch_size

        logger.info(f"Caching reference logits for {len(prompts)} prompts (top-k={top_k})...")

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_logits = self._forward_batch(batch_prompts)

            for logits in batch_logits:
                # logits: (seq_len, vocab_size)
                log_probs = F.log_softmax(logits.float(), dim=-1)

                # Take top-k per position
                top_values, top_indices = log_probs.topk(top_k, dim=-1)
                # Store on CPU: (seq_len, top_k)
                self.cached_references.append(
                    (top_values.cpu(), top_indices.cpu())
                )

            # Free GPU cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Cached reference logits for {len(self.cached_references)} prompts")

    def compute_kl_divergence(self, prompts: list[str], multiplier: float) -> KLResult:
        """
        Compute KL(P_orig || P_abl) using top-k approximation.

        Args:
            prompts: Same reference prompts used for caching (in same order)
            multiplier: The direction_multiplier that was used for abliteration

        Returns:
            KLResult with mean/median/max/std KL and per-prompt breakdown
        """
        if not self.cached_references:
            raise RuntimeError("No cached references. Call cache_reference_logits() first.")

        if len(prompts) != len(self.cached_references):
            raise ValueError(
                f"Prompt count mismatch: {len(prompts)} prompts vs "
                f"{len(self.cached_references)} cached references"
            )

        batch_size = self.config.batch_size
        per_prompt_kl = []
        prompt_idx = 0

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_logits = self._forward_batch(batch_prompts)

            for logits in batch_logits:
                if prompt_idx >= len(self.cached_references):
                    break

                ref_log_probs, ref_indices = self.cached_references[prompt_idx]

                # logits: (seq_len, vocab_size)
                abl_log_probs = F.log_softmax(logits.float(), dim=-1)

                # Match sequence lengths (use minimum)
                min_len = min(ref_log_probs.size(0), abl_log_probs.size(0))
                ref_lp = ref_log_probs[:min_len]  # (min_len, top_k)
                ref_idx = ref_indices[:min_len]  # (min_len, top_k)
                abl_lp = abl_log_probs[:min_len]  # (min_len, vocab)

                # Gather abliterated log probs at reference top-k indices
                abl_at_ref = abl_lp.gather(1, ref_idx)  # (min_len, top_k)

                # KL = sum_i P_ref(i) * (log P_ref(i) - log P_abl(i))
                # P_ref(i) = exp(ref_lp)
                p_ref = ref_lp.exp()
                kl_per_position = (p_ref * (ref_lp - abl_at_ref)).sum(dim=-1)  # (min_len,)

                # Clamp negative values (numerical artifacts from top-k approximation)
                kl_per_position = kl_per_position.clamp(min=0.0)

                # Average across positions for per-prompt KL
                prompt_kl = kl_per_position.mean().item()
                per_prompt_kl.append(prompt_kl)
                prompt_idx += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate statistics
        if not per_prompt_kl:
            raise RuntimeError("No per-prompt KL values computed. Check reference prompts and model state.")
        kl_tensor = torch.tensor(per_prompt_kl)
        return KLResult(
            mean_kl=kl_tensor.mean().item(),
            median_kl=kl_tensor.median().item(),
            max_kl=kl_tensor.max().item(),
            std_kl=kl_tensor.std().item() if len(kl_tensor) > 1 else 0.0,
            per_prompt_kl=per_prompt_kl,
            multiplier=multiplier,
        )


# ==============================================================================
# Auto-Tune Multiplier (Binary Search)
# ==============================================================================


def auto_tune_multiplier(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    directions,  # RefusalDirections
    config,  # AbliterationConfig
    kl_monitor: KLDivergenceMonitor,
    kl_config: KLMonitorConfig,
    reference_prompts: list[str],
    null_space_projector=None,
) -> AutoTuneResult:
    """
    Binary search over direction_multiplier to find the highest value
    that keeps KL divergence below the threshold.

    Each iteration: restore weights → abliterate with candidate → measure KL.
    Final state: model abliterated with the best multiplier found.

    Args:
        model: The model (pre-abliteration weights expected)
        tokenizer: Tokenizer
        directions: Computed RefusalDirections
        config: AbliterationConfig (modified in-place with best multiplier)
        kl_monitor: KLDivergenceMonitor with cached reference logits
        kl_config: KLMonitorConfig with search parameters
        reference_prompts: Prompts for KL measurement
        null_space_projector: Optional null-space projector

    Returns:
        AutoTuneResult with best multiplier, search history, convergence status
    """
    from src.abliterate import abliterate_model, get_linear_layer_names

    linear_names = get_linear_layer_names(model)

    # Create weight snapshot before any abliteration
    logger.info("Creating weight snapshot for auto-tune rollback...")
    snapshot = WeightSnapshot(model, linear_names)

    low = kl_config.search_min
    high = kl_config.search_max
    best_multiplier = low
    best_kl = float("inf")
    search_history = []
    converged = False

    logger.info(f"Auto-tune: searching [{low:.3f}, {high:.3f}], threshold={kl_config.kl_threshold:.3f}")

    for iteration in range(kl_config.max_search_iterations):
        candidate = (low + high) / 2.0
        logger.info(f"  Iteration {iteration + 1}: trying multiplier={candidate:.4f}")

        # Restore original weights
        snapshot.restore(model)

        # Abliterate with candidate multiplier
        original_multiplier = config.direction_multiplier
        config.direction_multiplier = candidate
        abliterate_model(model, directions, config, null_space_projector)
        config.direction_multiplier = original_multiplier  # Restore config

        # Measure KL
        kl_result = kl_monitor.compute_kl_divergence(reference_prompts, candidate)
        mean_kl = kl_result.mean_kl

        search_history.append({
            "iteration": iteration + 1,
            "multiplier": candidate,
            "mean_kl": mean_kl,
            "median_kl": kl_result.median_kl,
            "max_kl": kl_result.max_kl,
        })

        logger.info(f"    KL: mean={mean_kl:.4f}, median={kl_result.median_kl:.4f}, max={kl_result.max_kl:.4f}")

        if mean_kl < kl_config.kl_threshold:
            # KL within budget — try higher multiplier
            best_multiplier = candidate
            best_kl = mean_kl
            low = candidate
            logger.info(f"    Within budget. Best so far: {best_multiplier:.4f}")
        else:
            # KL too high — try lower multiplier
            high = candidate
            logger.info(f"    Over budget. Reducing range.")

        # Check convergence
        if (high - low) < kl_config.search_tolerance:
            converged = True
            logger.info(f"  Converged at iteration {iteration + 1} (range={high - low:.6f})")
            break

    num_iterations = len(search_history)

    if best_kl == float("inf"):
        logger.warning(
            f"No candidate met the KL threshold ({kl_config.kl_threshold}). "
            f"Using search_min={best_multiplier:.4f}. "
            "Consider lowering kl_search_min or raising kl_threshold."
        )

    # Final application: restore and abliterate with best multiplier
    logger.info(f"Applying best multiplier: {best_multiplier:.4f} (KL={best_kl:.4f})")
    snapshot.restore(model)
    config.direction_multiplier = best_multiplier
    abliterate_model(model, directions, config, null_space_projector)

    # Free snapshot
    snapshot.free()

    return AutoTuneResult(
        best_multiplier=best_multiplier,
        best_kl=best_kl,
        search_history=search_history,
        converged=converged,
        num_iterations=num_iterations,
        kl_threshold=kl_config.kl_threshold,
    )


# ==============================================================================
# Utilities
# ==============================================================================


def load_reference_prompts(
    path: Optional[str] = None,
    num_prompts: int = 50,
) -> list[str]:
    """
    Load reference prompts for KL monitoring.

    Falls back to preservation.txt from the null-space module.

    Args:
        path: Path to reference prompts file (one per line or JSON)
        num_prompts: Maximum number of prompts to use

    Returns:
        List of reference prompt strings
    """
    import random
    from src.abliterate import load_prompts_from_file

    if path is not None:
        prompts = load_prompts_from_file(path, num_prompts=None)
    else:
        # Default: use preservation prompts
        try:
            from src.null_space import get_default_preservation_prompts_path
            preservation_path = get_default_preservation_prompts_path()
            prompts = load_prompts_from_file(preservation_path, num_prompts=None)
        except (ImportError, FileNotFoundError):
            # Fallback: use harmless prompts
            from src.abliterate import get_default_prompts_path
            harmless_path = get_default_prompts_path("harmless.txt")
            prompts = load_prompts_from_file(harmless_path, num_prompts=None)

    # Sample if we have more than needed
    if len(prompts) > num_prompts:
        prompts = random.sample(prompts, num_prompts)

    return prompts


def save_kl_report(
    output_path: Path,
    kl_result: Optional[KLResult] = None,
    auto_tune_result: Optional[AutoTuneResult] = None,
):
    """Save KL divergence report to JSON file."""
    report = {}

    if kl_result is not None:
        report.update(kl_result.to_dict())

    if auto_tune_result is not None:
        report["auto_tune"] = auto_tune_result.to_dict()

    report_path = output_path / "kl_divergence_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved KL divergence report to {report_path}")
