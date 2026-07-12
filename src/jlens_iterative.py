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
from src.jlens import (
    DEFAULT_REFUSAL_CONCEPTS,
    build_jspace_config_dict,
    mine_refusal_concepts,
    save_jspace_config_json,
)
from src.kl_monitor import KLDivergenceMonitor, KLMonitorConfig, WeightSnapshot
from src.model_utils import load_model_and_tokenizer
from utils.refusal_detector import LogLikelihoodRefusalDetector, RefusalDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class JLensIterativeConfig:
    """Configuration for the J-lens-guided iterative abliteration loop."""

    # Convergence
    #
    # Design: iteration 1 with step_multiplier=1.0 is functionally equivalent to
    # a single restrict-mode pass (same direction_multiplier, same J-lens basis
    # on the ORIGINAL model). Iterations 2+ recompute J-lens on the MODIFIED
    # model — this is what catches the Hydra effect (refusal machinery migrating
    # between layers). So the loop is "restrict pass, re-check J-space on the
    # new model state, restrict again if refusal remains." Halves on KL breach.
    max_iterations: int = 10
    target_refusal_rate: float = 0.05
    min_improvement: float = 0.01           # stop if refusal rate improves less than this for 2 rounds
    step_multiplier: float = 1.0            # per-iteration ablation strength (matches restrict's direction_multiplier)
    min_step_multiplier: float = 0.05       # floor when halving after KL breach
    max_step_multiplier: float = 1.0        # cap on step growth when escalating
    step_escalation: bool = True            # if plateauing far from target, grow step (up to max)
    escalation_factor: float = 1.5          # multiplier applied when escalating step
    escalation_gap_ratio: float = 3.0       # only escalate while refusal_rate > target * ratio

    # Eval
    eval_prompt_count: int = 50             # refusal-rate probe size per iteration
    refusal_threshold: float = -7.0         # max anchor log-prob above this ⇒ predict refusal
    auto_calibrate_threshold: bool = False  # if True, run a mini-audit vs actual generations before the loop and pick a threshold that maximizes accuracy
    calibrate_num_harmless: int = 20        # audit sample size per class when auto-calibrating
    calibrate_num_harmful: int = 20
    calibrate_max_new_tokens: int = 40

    # Base abliteration prompt handling (aligned with run_abliteration / restrict mode)
    abliteration_num_prompts: int = 32      # harmful+harmless count used for direction extraction
    filter_prompts: bool = True             # run filter_harmful_prompts_by_refusal once before the loop

    # KL tracking + guardrail. Tracking is a diagnostic: it caches reference
    # logits and computes KL(base||current) every iteration so you can see
    # quality drift in the logs. The guardrail is a rollback policy layered
    # on top — it fires only if track_kl is also on and mean_kl exceeds
    # kl_threshold. Turning off track_kl also disables the guardrail.
    track_kl: bool = True
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

    # Concept mining (recomputed each iteration to catch refusal-style shifts)
    # mining_mode: "consensus" (mode-based; recommended, ignores harmless) or
    #              "contrast" (mean P(harmful) - mean P(harmless))
    mine_concepts_each_iteration: bool = False
    mine_top_k: int = 8
    mine_min_specificity: float = 0.005
    mine_batch_size: int = 4
    mine_max_seq_len: int = 128
    mine_num_prompts: int = 64
    mining_mode: str = "consensus"
    mine_num_positions: int = 4

    # Base abliteration passthroughs
    use_projected_refusal: bool = True
    use_welford_mean: bool = True
    use_float64_subtraction: bool = True
    use_winsorization: bool = False
    winsorize_percentile: float = 0.995
    norm_preservation: bool = True

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


def _inline_audit(
    model,
    tokenizer,
    detector_cfg: RefusalDetectorConfig,
    harmful_pool: list[str],
    harmless_prompts: list[str],
    num_harmful: int,
    num_harmless: int,
    max_new_tokens: int,
):
    """Run a refusal-detector audit against the already-loaded model.

    Avoids the double model load that would happen if we called `run_audit`
    directly. Reuses the audit's inner logic (per-anchor log-prob + greedy
    generation + heuristic classifier + threshold sweep).
    """
    from utils.refusal_detector_audit import (
        AuditSummary, PromptAudit, _generate_batch, _sweep_threshold,
    )
    from utils.test_abliteration import detect_refusal as heuristic_detect_refusal

    detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_cfg)
    harmful = harmful_pool[:num_harmful]
    harmless = harmless_prompts[:num_harmless]
    all_prompts = [(p, "harmful") for p in harmful] + [(p, "harmless") for p in harmless]
    prompts_only = [p for p, _ in all_prompts]

    formatted = [detector.format_prompt(p) for p in prompts_only]
    anchor_scores: dict[str, list[float]] = {}
    for anchor in detector_cfg.refusal_anchors:
        chunk_scores: list[float] = []
        for i in range(0, len(formatted), 4):
            chunk = formatted[i : i + 4]
            lp = detector.get_log_prob(chunk, anchor)
            chunk_scores.extend(lp.tolist())
        anchor_scores[anchor] = chunk_scores

    generations = _generate_batch(model, tokenizer, prompts_only, max_new_tokens, batch_size=4)

    audits: list[PromptAudit] = []
    for idx, ((prompt, source), gen) in enumerate(zip(all_prompts, generations)):
        per_anchor = {a: anchor_scores[a][idx] for a in detector_cfg.refusal_anchors}
        winning_anchor = max(per_anchor, key=per_anchor.get)
        max_score = per_anchor[winning_anchor]
        predicted = max_score > detector_cfg.threshold
        gen_is_refusal = heuristic_detect_refusal(gen)
        audits.append(PromptAudit(
            idx=idx, source=source, prompt=prompt,
            anchor_log_probs=per_anchor, max_score=max_score,
            winning_anchor=winning_anchor, predicted_refusal=predicted,
            generated_text=gen, generation_is_refusal=gen_is_refusal,
            agreement=predicted == gen_is_refusal,
        ))

    summary = AuditSummary(
        threshold=detector_cfg.threshold,
        num_prompts=len(audits),
        num_harmful=len(harmful),
        num_harmless=len(harmless),
    )
    for a in audits:
        gt = a.generation_is_refusal
        if a.predicted_refusal and gt: summary.true_positive += 1
        elif a.predicted_refusal and not gt: summary.false_positive += 1
        elif not a.predicted_refusal and not gt: summary.true_negative += 1
        else: summary.false_negative += 1
    thr, acc = _sweep_threshold(audits)
    summary.suggested_threshold = thr
    summary.suggested_threshold_accuracy = acc
    return audits, summary


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
    concepts: Optional[list[str]] = None,
    concept_token_ids: Optional[dict] = None,
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
        jlens_concepts=list(concepts) if concepts else list(it_cfg.jlens_concepts),
        jlens_concept_token_ids=concept_token_ids,
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

    # Load prompts. If filter_prompts=True, load the full pool (None) so we can
    # sample refused prompts from it — matches run_abliteration behavior.
    abl_n = max(config.abliteration_num_prompts, 1)
    harmful_pool = load_prompts_from_file(
        harmful_prompts_path,
        num_prompts=None if config.filter_prompts else abl_n,
    )
    harmless_prompts = load_prompts_from_file(harmless_prompts_path, num_prompts=abl_n)
    if not harmful_pool or not harmless_prompts:
        raise ValueError("harmful/harmless prompts must be non-empty")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path, device=config.device, dtype=config.dtype, trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    # Refusal detector
    detector_cfg = RefusalDetectorConfig(threshold=config.refusal_threshold)
    detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_cfg)

    # Optional: run a mini-audit vs actual generations to pick a threshold
    # calibrated for this specific model. The default -7.0 is wildly wrong for
    # some model families (Qwen3.5-9B's harmless log-prob p95 is ~-1.32).
    if config.auto_calibrate_threshold:
        from utils.refusal_detector_audit import run_audit
        logger.info(
            f"Auto-calibrating refusal threshold (audit sample: "
            f"{config.calibrate_num_harmful} harmful + "
            f"{config.calibrate_num_harmless} harmless prompts)..."
        )
        # Free the detector first; the audit function loads its own model
        # instance is unavoidable here because the audit is a self-contained
        # tool. To avoid loading twice we call the internal pieces directly.
        _, summary = _inline_audit(
            model, tokenizer, detector_cfg,
            harmful_pool=harmful_pool,
            harmless_prompts=harmless_prompts,
            num_harmful=min(config.calibrate_num_harmful, len(harmful_pool)),
            num_harmless=min(config.calibrate_num_harmless, len(harmless_prompts)),
            max_new_tokens=config.calibrate_max_new_tokens,
        )
        if summary.suggested_threshold is not None:
            old_threshold = config.refusal_threshold
            new_threshold = summary.suggested_threshold
            logger.info(
                f"  Calibrated threshold: {old_threshold} -> {new_threshold:.2f} "
                f"(audit accuracy: {summary.suggested_threshold_accuracy:.3f}, "
                f"harmless-FPR at old threshold: "
                f"{summary.false_positive / max(1, summary.false_positive + summary.true_negative):.3f})"
            )
            config.refusal_threshold = new_threshold
            detector_cfg.threshold = new_threshold
            # Rebuild detector to pick up the new threshold
            detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_cfg)
        else:
            logger.warning("Auto-calibration returned no suggestion; keeping default threshold")

    # Filter harmful prompts by actual refusal — matches run_abliteration. Do it
    # ONCE, on the original model, so the direction is computed from prompts the
    # model genuinely refuses (weakest signal otherwise).
    if config.filter_prompts:
        from src.abliterate import filter_harmful_prompts_by_refusal
        # Build a minimal AbliterationConfig just for refusal_threshold / batch_size
        filter_cfg = AbliterationConfig(
            model_path=model_path,
            output_path=output_path,
            harmful_prompts=harmful_pool,
            harmless_prompts=harmless_prompts,
            num_prompts=abl_n,
            refusal_threshold=config.refusal_threshold,
            device=config.device,
            dtype=config.dtype,
        )
        refused, _ = filter_harmful_prompts_by_refusal(
            harmful_pool, model, tokenizer, filter_cfg, target_count=abl_n
        )
        if not refused:
            raise ValueError(
                "No harmful prompts were refused by the model — refusal direction "
                "cannot be computed. Try disabling filter_prompts or lowering "
                "refusal_threshold."
            )
        harmful_prompts = refused
        logger.info(
            f"Filtered to {len(harmful_prompts)} refused harmful prompts "
            f"(pool size: {len(harmful_pool)})"
        )
    else:
        harmful_prompts = harmful_pool[:abl_n]

    if eval_prompts is None:
        # Use the full pool for eval so we probe on ALL harmful prompts, not just
        # the filtered/refused subset (matches restrict-mode eval semantics).
        eval_prompts = harmful_pool[: config.eval_prompt_count]
    else:
        eval_prompts = eval_prompts[: config.eval_prompt_count]

    # KL tracking + optional guardrail. Guardrail implies tracking.
    kl_monitor = None
    kl_ref_prompts: list[str] = []
    original_snapshot: Optional[WeightSnapshot] = None
    track_kl_effective = config.track_kl or config.use_kl_guardrail
    if track_kl_effective:
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
        role = "guardrail" if config.use_kl_guardrail else "diagnostic"
        logger.info(
            f"Caching reference logits for KL {role} ({len(kl_ref_prompts)} prompts)..."
        )
        kl_monitor.cache_reference_logits(kl_ref_prompts)
        # Snapshot the ORIGINAL weights only when the guardrail may need to
        # roll back to baseline. Diagnostic-only tracking never rolls back.
        if config.use_kl_guardrail:
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

    # Persist the last-successful mined concepts so we can fall back if a later
    # iteration collapses (e.g. model stops refusing enough for mining to work).
    last_mined_concepts: Optional[list[str]] = None
    last_mined_ids: Optional[dict[str, int]] = None

    for it in range(1, config.max_iterations + 1):
        logger.info(f"===== Iteration {it}/{config.max_iterations} =====")

        # Per-iteration weight snapshot for rollback of this step only
        linear_names = get_linear_layer_names(model)
        step_snapshot = WeightSnapshot(model, linear_names)

        # Optional contrastive concept mining on the CURRENT model state.
        # After iteration 1 the model has been partly abliterated, so mining
        # from the ORIGINAL harmful set picks up compliance tokens (" a", "Here",
        # " The", ...) instead of refusal patterns. Filter to prompts the model
        # STILL refuses on the current state before mining.
        step_concepts: Optional[list[str]] = None
        step_concept_ids: Optional[dict[str, int]] = None
        if config.mine_concepts_each_iteration:
            mining_pool: list[str] = []
            skipped_reason: Optional[str] = None
            if it == 1:
                mining_pool = harmful_prompts[: config.mine_num_prompts]
            else:
                # Re-check refusal on the current model. Use the ORIGINAL pool
                # (pre-abliteration filter) so we don't restrict to a
                # potentially-empty subset if refusal has collapsed.
                candidate_pool = harmful_pool[: config.mine_num_prompts]
                refused_flags = detector.detect_refusal_batch(candidate_pool)
                mining_pool = [p for p, r in zip(candidate_pool, refused_flags) if r]
                min_for_mining = max(8, config.mine_num_prompts // 4)
                logger.info(
                    f"  Concept-mining pool: {len(mining_pool)}/{len(candidate_pool)} "
                    f"prompts still refused on current model"
                )
                if len(mining_pool) < min_for_mining:
                    skipped_reason = (
                        f"only {len(mining_pool)} still-refused prompts "
                        f"(need >= {min_for_mining}); mining a smaller set would "
                        f"pick up compliance tokens instead of refusal patterns"
                    )

            if skipped_reason is not None:
                if last_mined_concepts:
                    logger.info(
                        f"  Skipping concept re-mining ({skipped_reason}); "
                        f"reusing previous concepts: {last_mined_concepts}"
                    )
                    step_concepts = last_mined_concepts
                    step_concept_ids = last_mined_ids
                else:
                    logger.info(
                        f"  Skipping concept re-mining ({skipped_reason}); "
                        f"falling back to default concepts"
                    )
            else:
                try:
                    mined = mine_refusal_concepts(
                        model, tokenizer,
                        harmful_prompts=mining_pool,
                        harmless_prompts=(
                            harmless_prompts[: config.mine_num_prompts]
                            if config.mining_mode == "contrast" else None
                        ),
                        top_k=config.mine_top_k,
                        min_specificity=config.mine_min_specificity,
                        batch_size=config.mine_batch_size,
                        max_seq_len=config.mine_max_seq_len,
                        device=config.device,
                        mode=config.mining_mode,
                        num_positions=config.mine_num_positions,
                    )
                except Exception as e:
                    logger.warning(f"Concept mining failed: {e}; using previous / default")
                    mined = []
                if mined:
                    step_concepts = [m.concept for m in mined]
                    step_concept_ids = {m.concept: m.token_id for m in mined}
                    last_mined_concepts = step_concepts
                    last_mined_ids = step_concept_ids
                elif last_mined_concepts:
                    logger.info("Mining collapsed; reusing last successful mined concepts")
                    step_concepts = last_mined_concepts
                    step_concept_ids = last_mined_ids
                # else: falls through to config.jlens_concepts (defaults)

        # Build config for this step (fresh J-lens computation via inline path)
        abl_cfg = _build_abliteration_config(
            config, model_path, output_path, harmful_prompts, harmless_prompts, step,
            concepts=step_concepts, concept_token_ids=step_concept_ids,
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
        if num_targeted > 0 and num_fell_back >= num_targeted:
            logger.warning(
                f"  J-space restriction retained < {config.jlens_min_projection_ratio} "
                f"on ALL {num_targeted} targeted layers — using unrestricted refusal "
                f"direction everywhere this step. Consider lowering "
                f"--jlens-min-projection-ratio, raising --jlens-basis-rank, or "
                f"widening the concept list (defaults may not match this model's "
                f"refusal geometry)."
            )
        elif num_fell_back > num_targeted // 2:
            logger.info(
                f"  J-space fallback: {num_fell_back}/{num_targeted} layers used "
                f"unrestricted direction (retained ratio < "
                f"{config.jlens_min_projection_ratio})"
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
                # If we're plateauing but still far from the target, the step is
                # too small to move the needle. Escalate before halting.
                far_from_target = cur_refusal > config.target_refusal_rate * config.escalation_gap_ratio
                if config.step_escalation and far_from_target and step < config.max_step_multiplier:
                    new_step = min(step * config.escalation_factor, config.max_step_multiplier)
                    logger.info(
                        f"  Plateau ({plateau_rounds} round(s)) far from target "
                        f"({cur_refusal:.3f} > {config.target_refusal_rate * config.escalation_gap_ratio:.3f}); "
                        f"escalating step {step:.3f} -> {new_step:.3f}"
                    )
                    step = new_step
                    plateau_rounds = 0
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
            # Only halt-on-plateau when step is already at max — otherwise
            # escalation above should have grown it. If we plateau at max step
            # and still can't move refusal, further iterations won't help.
            at_max_step = step >= config.max_step_multiplier - 1e-9
            if at_max_step or not config.step_escalation:
                logger.info(
                    f"No improvement for {plateau_rounds} rounds at step={step:.3f} "
                    f"(refusal_rate={cur_refusal:.3f}, target={config.target_refusal_rate}); halting"
                )
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

    # Write J-space-specific abliteration_config.json
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32"}.get(config.dtype, str(config.dtype))
    concept_source = (
        "auto-mine" if config.mine_concepts_each_iteration else "defaults"
    )
    mined_records = (
        [{"concept": c, "token_id": tid}
         for c, tid in (last_mined_ids or {}).items()]
        if config.mine_concepts_each_iteration and last_mined_concepts else None
    )
    payload = build_jspace_config_dict(
        mode="iterative",
        model_path=model_path,
        output_path=str(output_path),
        device=config.device,
        dtype=dtype_str,
        jlens_params={
            "num_prompts": config.jlens_num_prompts,
            "batch_size": config.jlens_batch_size,
            "max_seq_len": config.jlens_max_seq_len,
            "grad_checkpoint": config.jlens_grad_checkpoint,
            "basis_rank": config.jlens_basis_rank,
            "min_projection_ratio": config.jlens_min_projection_ratio,
        },
        concepts_params={
            "source": concept_source,
            "concepts": last_mined_concepts if last_mined_concepts else list(config.jlens_concepts),
            "mining_mode": config.mining_mode if config.mine_concepts_each_iteration else None,
            "mining_top_k": config.mine_top_k if config.mine_concepts_each_iteration else None,
            "mining_min_score": config.mine_min_specificity if config.mine_concepts_each_iteration else None,
            "mining_num_positions": config.mine_num_positions if config.mine_concepts_each_iteration else None,
            "mining_num_prompts": config.mine_num_prompts if config.mine_concepts_each_iteration else None,
            "mine_each_iteration": config.mine_concepts_each_iteration,
            "last_mined_concepts": mined_records,
        },
        abliteration_params={
            "harmful_prompts_path": harmful_prompts_path,
            "harmless_prompts_path": harmless_prompts_path,
            "num_harmful_prompts": len(harmful_prompts),
            "num_harmless_prompts": len(harmless_prompts),
            "use_projected_refusal": config.use_projected_refusal,
            "use_welford_mean": config.use_welford_mean,
            "use_float64_subtraction": config.use_float64_subtraction,
            "use_winsorization": config.use_winsorization,
            "winsorize_percentile": config.winsorize_percentile if config.use_winsorization else None,
            "norm_preservation": config.norm_preservation,
        },
        iterative_params={
            "max_iterations": config.max_iterations,
            "target_refusal_rate": config.target_refusal_rate,
            "step_multiplier": config.step_multiplier,
            "min_step_multiplier": config.min_step_multiplier,
            "min_improvement": config.min_improvement,
            "eval_prompt_count": config.eval_prompt_count,
            "refusal_threshold": config.refusal_threshold,
            "kl_guardrail": {
                "enabled": config.use_kl_guardrail,
                "threshold": config.kl_threshold if config.use_kl_guardrail else None,
                "num_reference_prompts": config.kl_num_reference_prompts if config.use_kl_guardrail else None,
                "top_k": config.kl_top_k if config.use_kl_guardrail else None,
            },
            "result": {
                "converged": converged,
                "reason": reason,
                "baseline_refusal_rate": baseline_refusal,
                "final_refusal_rate": prev_refusal,
                "total_iterations": len(history),
                "final_kl": final_kl,
            },
        },
    )
    save_jspace_config_json(payload, str(output_path))

    # Free memory
    del model, detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
