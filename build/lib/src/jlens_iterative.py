"""J-lens-guided iterative abliteration.

Loop:
  1. Compute J-lens vectors on the CURRENT (possibly already-modified) model.
     Recomputing per iteration is the point — it detects refusal machinery
     migrating between layers (Hydra effect / self-repair).
  2. Compute refusal directions with J-space restriction.
  3. Snapshot linear-layer weights.
  4. Apply a single-step abliteration (small multiplier).
  5. Measure refusal rate (LogLikelihood detector) + optional KL from ORIGINAL model.
  6. If KL guardrail exceeded OR refusal rate got worse -> rollback + stop.
  7. Converge when refusal_rate <= target OR max_iterations.

Architecture-agnostic: unlike SAE-guided iterative modes, requires no pretrained
SAEs — just autograd + the log-likelihood detector.
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.abliterate import (
    AbliterationConfig,
    abliterate_model,
    compute_refusal_directions,
    get_linear_layer_names,
    load_prompts_from_file,
)
from src.jlens import DEFAULT_REFUSAL_CONCEPTS
from src.kl_monitor import KLDivergenceMonitor, KLMonitorConfig, WeightSnapshot
from src.model_utils import load_model_and_tokenizer
from utils.refusal_detector import LogLikelihoodRefusalDetector, RefusalDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class JLensIterativeConfig:
    """Configuration for the J-lens-guided iterative abliteration loop."""

    # Convergence
    max_iterations: int = 10
    target_refusal_rate: float = 0.05
    min_improvement: float = 0.01           # stop if refusal rate improves less than this for 2 rounds
    step_multiplier: float = 0.4            # per-iteration ablation strength
    min_step_multiplier: float = 0.05       # floor when halving after KL breach

    # Eval
    eval_prompt_count: int = 50             # refusal-rate probe size per iteration
    refusal_threshold: float = -7.0

    # KL guardrail (optional)
    use_kl_guardrail: bool = False
    kl_threshold: float = 0.5
    kl_reference_prompts_path: Optional[str] = None
    kl_num_reference_prompts: int = 30
    kl_top_k: int = 200
    kl_batch_size: int = 4

    # J-lens extraction (per iteration)
    jlens_concepts: list[str] = field(default_factory=lambda: list(DEFAULT_REFUSAL_CONCEPTS))
    jlens_num_prompts: int = 32
    jlens_batch_size: int = 2
    jlens_max_seq_len: int = 64
    jlens_grad_checkpoint: bool = False
    jlens_basis_rank: int = 16
    jlens_min_projection_ratio: float = 0.1

    # Base abliteration passthroughs
    use_projected_refusal: bool = True
    use_welford_mean: bool = True
    use_float64_subtraction: bool = True
    use_winsorization: bool = False
    winsorize_percentile: float = 0.995
    norm_preservation: bool = True
    filter_prompts: bool = True

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16


@dataclass
class IterationState:
    """Per-iteration snapshot for the loop's decision-making."""

    iteration: int
    step_multiplier: float
    refusal_rate: float
    prev_refusal_rate: float
    mean_kl: Optional[float]
    max_kl: Optional[float]
    num_layers_targeted: int
    fell_back: int              # layers where J-lens restriction fell back to unrestricted
    action: str                 # "applied", "rolled_back_kl", "rolled_back_no_improvement", "converged", "halted"


@dataclass
class IterationResult:
    """Final result of the iterative loop."""

    converged: bool
    reason: str
    final_refusal_rate: float
    baseline_refusal_rate: float
    total_iterations: int
    history: list[IterationState]
    final_kl: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "reason": self.reason,
            "final_refusal_rate": self.final_refusal_rate,
            "baseline_refusal_rate": self.baseline_refusal_rate,
            "total_iterations": self.total_iterations,
            "final_kl": self.final_kl,
            "history": [asdict(s) for s in self.history],
        }


def _measure_refusal_rate(
    detector: LogLikelihoodRefusalDetector, prompts: list[str]
) -> float:
    """Fraction of prompts the model would refuse, per the log-likelihood detector."""
    if not prompts:
        return 0.0
    results = detector.detect_refusal_batch(prompts)
    return sum(1 for r in results if r) / len(results)


def _build_abliteration_config(
    it_cfg: JLensIterativeConfig,
    model_path: str,
    output_path: str,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    step_multiplier: float,
) -> AbliterationConfig:
    """Assemble an AbliterationConfig with J-lens restriction enabled for one iteration step."""
    return AbliterationConfig(
        model_path=model_path,
        output_path=output_path,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        num_prompts=len(harmful_prompts),
        direction_multiplier=step_multiplier,
        norm_preservation=it_cfg.norm_preservation,
        device=it_cfg.device,
        dtype=it_cfg.dtype,
        use_projected_refusal=it_cfg.use_projected_refusal,
        use_welford_mean=it_cfg.use_welford_mean,
        use_float64_subtraction=it_cfg.use_float64_subtraction,
        use_winsorization=it_cfg.use_winsorization,
        winsorize_percentile=it_cfg.winsorize_percentile,
        filter_harmful_prompts=False,  # done once up-front
        # J-lens restriction: computed fresh each iteration on the current model
        use_jlens_restriction=True,
        jlens_vectors_path=None,
        jlens_basis_rank=it_cfg.jlens_basis_rank,
        jlens_min_projection_ratio=it_cfg.jlens_min_projection_ratio,
        jlens_num_prompts=it_cfg.jlens_num_prompts,
        jlens_batch_size=it_cfg.jlens_batch_size,
        jlens_max_seq_len=it_cfg.jlens_max_seq_len,
        jlens_grad_checkpoint=it_cfg.jlens_grad_checkpoint,
        jlens_concepts=list(it_cfg.jlens_concepts),
    )


def run_jlens_iterative_abliteration(
    model_path: str,
    output_path: str,
    harmful_prompts_path: str,
    harmless_prompts_path: str,
    config: JLensIterativeConfig,
    eval_prompts: Optional[list[str]] = None,
    kl_reference_prompts: Optional[list[str]] = None,
) -> IterationResult:
    """Main iterative abliteration loop.

    Persists to `output_path` on convergence. Also writes:
      - iteration_history.json  (per-step metrics)
      - abliteration_config.json (final effective config)
    """
    logger.info(f"J-lens iterative abliteration: {model_path} -> {output_path}")
    logger.info(
        f"  target_refusal_rate={config.target_refusal_rate}, "
        f"max_iterations={config.max_iterations}, step={config.step_multiplier}"
    )

    # Load prompts
    harmful_prompts = load_prompts_from_file(
        harmful_prompts_path, num_prompts=max(config.jlens_num_prompts, 50)
    )
    harmless_prompts = load_prompts_from_file(
        harmless_prompts_path, num_prompts=max(config.jlens_num_prompts, 50)
    )
    if not harmful_prompts or not harmless_prompts:
        raise ValueError("harmful/harmless prompts must be non-empty")

    if eval_prompts is None:
        eval_prompts = harmful_prompts[: config.eval_prompt_count]
    else:
        eval_prompts = eval_prompts[: config.eval_prompt_count]

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path, device=config.device, dtype=config.dtype, trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    # Refusal detector
    detector_cfg = RefusalDetectorConfig(threshold=config.refusal_threshold)
    detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_cfg)

    # Optional KL guardrail
    kl_monitor = None
    kl_ref_prompts: list[str] = []
    original_snapshot: Optional[WeightSnapshot] = None
    if config.use_kl_guardrail:
        # Load reference prompts for KL
        if kl_reference_prompts is None:
            path = config.kl_reference_prompts_path
            if path is None:
                # Reuse harmless prompts if no preservation file supplied
                kl_ref_prompts = harmless_prompts[: config.kl_num_reference_prompts]
            else:
                kl_ref_prompts = load_prompts_from_file(path, num_prompts=config.kl_num_reference_prompts)
        else:
            kl_ref_prompts = kl_reference_prompts[: config.kl_num_reference_prompts]

        kl_config = KLMonitorConfig(
            num_reference_prompts=len(kl_ref_prompts),
            top_k=config.kl_top_k,
            batch_size=config.kl_batch_size,
            kl_threshold=config.kl_threshold,
        )
        kl_monitor = KLDivergenceMonitor(model, tokenizer, kl_config, device=config.device)
        logger.info(f"Caching reference logits for KL guardrail ({len(kl_ref_prompts)} prompts)...")
        kl_monitor.cache_reference_logits(kl_ref_prompts)
        # Snapshot the ORIGINAL weights so we can always roll back to baseline
        linear_names = get_linear_layer_names(model)
        original_snapshot = WeightSnapshot(model, linear_names)

    # Baseline refusal rate
    baseline_refusal = _measure_refusal_rate(detector, eval_prompts)
    logger.info(f"Baseline refusal rate: {baseline_refusal:.3f}")

    history: list[IterationState] = []
    prev_refusal = baseline_refusal
    step = config.step_multiplier
    final_kl: Optional[float] = None
    reason = "max_iterations reached"
    converged = False
    plateau_rounds = 0

    if baseline_refusal <= config.target_refusal_rate:
        logger.info("Baseline already meets target; no abliteration needed")
        return IterationResult(
            converged=True,
            reason="baseline_at_target",
            final_refusal_rate=baseline_refusal,
            baseline_refusal_rate=baseline_refusal,
            total_iterations=0,
            history=[],
            final_kl=0.0,
        )

    for it in range(1, config.max_iterations + 1):
        logger.info(f"===== Iteration {it}/{config.max_iterations} =====")

        # Per-iteration weight snapshot for rollback of this step only
        linear_names = get_linear_layer_names(model)
        step_snapshot = WeightSnapshot(model, linear_names)

        # Build config for this step (fresh J-lens computation via inline path)
        abl_cfg = _build_abliteration_config(
            config, model_path, output_path, harmful_prompts, harmless_prompts, step
        )

        try:
            directions = compute_refusal_directions(model, tokenizer, abl_cfg)
        except Exception as e:
            logger.error(f"Direction computation failed: {e}")
            step_snapshot.free()
            reason = f"direction_computation_error: {e}"
            break

        num_targeted = len(directions.directions)
        # Count fallbacks from metadata
        ratios = directions.metadata.get("jlens_restriction_ratios") or {}
        num_fell_back = sum(
            1 for r in ratios.values() if r < config.jlens_min_projection_ratio
        )

        # Apply abliteration
        abliterate_model(model, directions, abl_cfg, None)

        # Measure refusal rate
        cur_refusal = _measure_refusal_rate(detector, eval_prompts)
        # Optional KL from ORIGINAL model
        mean_kl: Optional[float] = None
        max_kl: Optional[float] = None
        if kl_monitor is not None:
            kl_result = kl_monitor.compute_kl_divergence(kl_ref_prompts, step)
            mean_kl = kl_result.mean_kl
            max_kl = kl_result.max_kl

        action = "applied"

        # KL guardrail
        if kl_monitor is not None and mean_kl is not None and mean_kl > config.kl_threshold:
            logger.warning(
                f"  KL {mean_kl:.3f} > threshold {config.kl_threshold}; rolling back this step"
            )
            step_snapshot.restore(model)
            action = "rolled_back_kl"
            # Halve step for next attempt
            step = max(step * 0.5, config.min_step_multiplier)
            cur_refusal = prev_refusal
        else:
            # If refusal got worse (or didn't improve), roll back and stop
            improvement = prev_refusal - cur_refusal
            if cur_refusal > prev_refusal + 0.01:
                logger.warning(
                    f"  Refusal rate got worse ({prev_refusal:.3f} -> {cur_refusal:.3f}); rolling back"
                )
                step_snapshot.restore(model)
                cur_refusal = prev_refusal
                action = "rolled_back_no_improvement"
                reason = "refusal_rate_regressed"
                step_snapshot.free()
                history.append(IterationState(
                    iteration=it, step_multiplier=step, refusal_rate=cur_refusal,
                    prev_refusal_rate=prev_refusal, mean_kl=mean_kl, max_kl=max_kl,
                    num_layers_targeted=num_targeted, fell_back=num_fell_back, action=action,
                ))
                final_kl = mean_kl
                break
            elif improvement < config.min_improvement:
                plateau_rounds += 1
            else:
                plateau_rounds = 0

        history.append(IterationState(
            iteration=it,
            step_multiplier=step,
            refusal_rate=cur_refusal,
            prev_refusal_rate=prev_refusal,
            mean_kl=mean_kl,
            max_kl=max_kl,
            num_layers_targeted=num_targeted,
            fell_back=num_fell_back,
            action=action,
        ))
        step_snapshot.free()
        final_kl = mean_kl

        logger.info(
            f"  refusal_rate: {prev_refusal:.3f} -> {cur_refusal:.3f} "
            f"(target={config.target_refusal_rate}), "
            f"kl_mean={mean_kl if mean_kl is not None else 'n/a'}, "
            f"targeted={num_targeted}, fell_back={num_fell_back}, action={action}"
        )

        prev_refusal = cur_refusal

        # Convergence checks
        if cur_refusal <= config.target_refusal_rate:
            logger.info(f"Converged at iteration {it} (refusal_rate={cur_refusal:.3f})")
            converged = True
            reason = "target_reached"
            break
        if plateau_rounds >= 2:
            logger.info(f"No improvement for {plateau_rounds} rounds; halting")
            reason = "plateau"
            break

    # Free KL monitor caches (no dedicated .free method, just drop references)
    if kl_monitor is not None:
        kl_monitor.cached_references.clear()
    if original_snapshot is not None:
        original_snapshot.free()

    # Save the (possibly modified) model to output_path
    logger.info(f"Saving abliterated model to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Persist iteration history
    result = IterationResult(
        converged=converged,
        reason=reason,
        final_refusal_rate=prev_refusal,
        baseline_refusal_rate=baseline_refusal,
        total_iterations=len(history),
        history=history,
        final_kl=final_kl,
    )
    with open(Path(output_path) / "iteration_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "model_path": model_path,
                "config": {k: (str(v) if isinstance(v, torch.dtype) else v)
                           for k, v in asdict(config).items()},
                "result": result.to_dict(),
            },
            f,
            indent=2,
            default=str,
        )

    # Free memory
    del model, detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
