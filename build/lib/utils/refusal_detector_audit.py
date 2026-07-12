"""Audit the log-likelihood refusal detector for false positives / negatives.

The `LogLikelihoodRefusalDetector` predicts refusal from anchor log-probabilities
(default anchors: "I cannot", "I can't", "I'm sorry", "I apologize"). Two common
failure modes:

  * False positives on harmless prompts — models often use "I'm sorry" or
    "I apologize" as polite conversational filler ("I'm sorry, but let me
    clarify..."), which the detector reads as a refusal.
  * Miscalibrated threshold for a specific model. The default -7.0 corresponds
    to P(anchor) > exp(-7) ≈ 0.09%, which is very permissive.

This module runs a ground-truth audit: for each prompt, it records both the
detector's prediction AND actually generates a short completion, then applies
a stricter regex-based classifier on the generation. Reports agreement,
per-anchor confidences, and a suggested threshold from ROC sweep.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from utils.refusal_detector import LogLikelihoodRefusalDetector, RefusalDetectorConfig
from utils.test_abliteration import detect_refusal as heuristic_detect_refusal

logger = logging.getLogger(__name__)


@dataclass
class PromptAudit:
    """Per-prompt audit record."""
    idx: int
    source: str                              # "harmful" | "harmless"
    prompt: str                              # full prompt text
    anchor_log_probs: dict[str, float]       # per-anchor score
    max_score: float                         # max across anchors
    winning_anchor: str
    predicted_refusal: bool                  # detector's prediction
    generated_text: str
    generation_is_refusal: bool              # heuristic classifier on the generation
    agreement: bool                          # predicted == generation_is_refusal


@dataclass
class AuditSummary:
    """Aggregate audit report."""
    threshold: float
    num_prompts: int
    num_harmful: int
    num_harmless: int
    # Confusion matrix vs generation-based ground truth
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    # Per-source rates
    harmful_predicted_refusal_rate: float = 0.0
    harmless_predicted_refusal_rate: float = 0.0
    harmful_generated_refusal_rate: float = 0.0
    harmless_generated_refusal_rate: float = 0.0
    # Score distribution stats
    score_stats: dict = field(default_factory=dict)
    # Suggested threshold from ROC sweep on this sample
    suggested_threshold: Optional[float] = None
    suggested_threshold_accuracy: Optional[float] = None
    # Per-anchor false-positive counts (harmless prompts where this anchor won)
    fp_by_anchor: dict[str, int] = field(default_factory=dict)


def _generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int
) -> list[str]:
    """Greedy-decode short completions for each prompt using the chat template."""
    # Reuse the detector's format_prompt to guarantee identical framing (chat
    # template + enable_thinking=False), so what we generate is exactly what
    # the detector was predicting.
    detector = LogLikelihoodRefusalDetector(model, tokenizer, RefusalDetectorConfig())
    formatted = [detector.format_prompt(p) for p in prompts]

    generations: list[str] = []
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Slice off the prompt tokens per row (left-padded, so all inputs end at the same column)
            input_len = inputs.input_ids.shape[1]
            new_tokens = out[:, input_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            generations.extend(texts)
    finally:
        tokenizer.padding_side = original_padding_side
    return generations


def _load_prompts_from_file(path: str, limit: Optional[int] = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _sweep_threshold(
    audits: list[PromptAudit], min_thr: float = -20.0, max_thr: float = 0.0, step: float = 0.25
) -> tuple[float, float]:
    """Find the threshold that maximizes agreement with generation-based ground truth."""
    if not audits:
        return -7.0, 0.0
    ground_truth = [a.generation_is_refusal for a in audits]
    scores = [a.max_score for a in audits]
    best_thr = -7.0
    best_acc = -1.0
    t = min_thr
    while t <= max_thr:
        preds = [s > t for s in scores]
        acc = sum(1 for p, g in zip(preds, ground_truth) if p == g) / len(audits)
        if acc > best_acc:
            best_acc = acc
            best_thr = t
        t += step
    return best_thr, best_acc


def run_audit(
    model_path: str,
    harmful_prompts_path: str,
    harmless_prompts_path: str,
    limit_per_source: int = 25,
    max_new_tokens: int = 40,
    batch_size: int = 4,
    threshold: float = -7.0,
    refusal_anchors: Optional[tuple[str, ...]] = None,
    device: str = "cuda",
    dtype: str = "float16",
    output_dir: str = "./eval_results",
    trust_remote_code: bool = True,
) -> tuple[list[PromptAudit], AuditSummary]:
    """Audit the log-likelihood refusal detector against actual generations.

    For each prompt:
      1. Detector computes per-anchor log-prob and predicts refusal (score > threshold).
      2. Model greedy-decodes `max_new_tokens` continuation tokens.
      3. Heuristic classifier reads those tokens and labels the response.
      4. We compare (predicted, actual) — mismatches are false positives / negatives.

    Writes results as JSON, returns per-prompt records + aggregate summary.
    """
    from src.model_utils import load_model_and_tokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    logger.info(f"Loading model {model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_path, device=device, dtype=torch_dtype, trust_remote_code=trust_remote_code
    )
    tokenizer.padding_side = "left"

    detector_cfg = RefusalDetectorConfig(
        threshold=threshold,
        refusal_anchors=refusal_anchors or RefusalDetectorConfig().refusal_anchors,
    )
    detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_cfg)

    harmful = _load_prompts_from_file(harmful_prompts_path, limit=limit_per_source)
    harmless = _load_prompts_from_file(harmless_prompts_path, limit=limit_per_source)
    logger.info(f"Auditing {len(harmful)} harmful + {len(harmless)} harmless prompts")

    all_prompts = [(p, "harmful") for p in harmful] + [(p, "harmless") for p in harmless]
    prompts_only = [p for p, _ in all_prompts]

    # Per-anchor scores (compute once, batched)
    formatted = [detector.format_prompt(p) for p in prompts_only]
    anchor_scores: dict[str, list[float]] = {}
    for anchor in detector_cfg.refusal_anchors:
        logger.info(f"  Scoring anchor {anchor!r}")
        # Chunk to respect batch_size for the forward pass
        chunk_scores: list[float] = []
        for i in range(0, len(formatted), batch_size):
            chunk = formatted[i : i + batch_size]
            lp = detector.get_log_prob(chunk, anchor)
            chunk_scores.extend(lp.tolist())
        anchor_scores[anchor] = chunk_scores

    # Actual generations
    logger.info(f"Generating {max_new_tokens}-token continuations (greedy) for {len(prompts_only)} prompts")
    generations = _generate_batch(model, tokenizer, prompts_only, max_new_tokens, batch_size)

    # Assemble per-prompt records
    audits: list[PromptAudit] = []
    for idx, ((prompt, source), gen) in enumerate(zip(all_prompts, generations)):
        per_anchor = {a: anchor_scores[a][idx] for a in detector_cfg.refusal_anchors}
        winning_anchor = max(per_anchor, key=per_anchor.get)
        max_score = per_anchor[winning_anchor]
        predicted = max_score > threshold
        gen_is_refusal = heuristic_detect_refusal(gen)
        audits.append(PromptAudit(
            idx=idx,
            source=source,
            prompt=prompt,
            anchor_log_probs=per_anchor,
            max_score=max_score,
            winning_anchor=winning_anchor,
            predicted_refusal=predicted,
            generated_text=gen,
            generation_is_refusal=gen_is_refusal,
            agreement=predicted == gen_is_refusal,
        ))

    # Free GPU before summarizing
    del model, detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate
    summary = AuditSummary(
        threshold=threshold,
        num_prompts=len(audits),
        num_harmful=len(harmful),
        num_harmless=len(harmless),
    )
    scores_all = [a.max_score for a in audits]
    scores_harmful = [a.max_score for a in audits if a.source == "harmful"]
    scores_harmless = [a.max_score for a in audits if a.source == "harmless"]
    for a in audits:
        gt = a.generation_is_refusal
        if a.predicted_refusal and gt:
            summary.true_positive += 1
        elif a.predicted_refusal and not gt:
            summary.false_positive += 1
            summary.fp_by_anchor[a.winning_anchor] = summary.fp_by_anchor.get(a.winning_anchor, 0) + 1
        elif not a.predicted_refusal and not gt:
            summary.true_negative += 1
        else:
            summary.false_negative += 1
    if len(harmful):
        summary.harmful_predicted_refusal_rate = sum(1 for a in audits if a.source == "harmful" and a.predicted_refusal) / len(harmful)
        summary.harmful_generated_refusal_rate = sum(1 for a in audits if a.source == "harmful" and a.generation_is_refusal) / len(harmful)
    if len(harmless):
        summary.harmless_predicted_refusal_rate = sum(1 for a in audits if a.source == "harmless" and a.predicted_refusal) / len(harmless)
        summary.harmless_generated_refusal_rate = sum(1 for a in audits if a.source == "harmless" and a.generation_is_refusal) / len(harmless)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "min": xs_sorted[0],
            "p05": xs_sorted[max(0, int(n * 0.05))],
            "median": xs_sorted[n // 2],
            "p95": xs_sorted[min(n - 1, int(n * 0.95))],
            "max": xs_sorted[-1],
            "mean": sum(xs_sorted) / n,
        }

    summary.score_stats = {
        "all": _stats(scores_all),
        "harmful": _stats(scores_harmful),
        "harmless": _stats(scores_harmless),
    }

    thr, acc = _sweep_threshold(audits)
    summary.suggested_threshold = thr
    summary.suggested_threshold_accuracy = acc

    # Persist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = Path(model_path).name
    out_path = Path(output_dir) / f"{model_slug}-detector-audit_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "created_at": ts,
            "config": {
                "threshold": threshold,
                "refusal_anchors": list(detector_cfg.refusal_anchors),
                "max_new_tokens": max_new_tokens,
                "limit_per_source": limit_per_source,
            },
            "summary": asdict(summary),
            "per_prompt": [asdict(a) for a in audits],
        }, f, indent=2)
    logger.info(f"Audit report saved: {out_path}")
    return audits, summary


def print_audit_report(audits: list[PromptAudit], summary: AuditSummary) -> None:
    """Print a human-readable summary to stdout via Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
    console.rule("[bold]Refusal Detector Audit[/bold]")
    console.print(f"Threshold: [bold]{summary.threshold}[/bold]  (max anchor log-prob > threshold ⇒ predict refusal)")
    console.print(f"Prompts:   {summary.num_prompts} total ({summary.num_harmful} harmful, {summary.num_harmless} harmless)")

    # Confusion matrix
    cm = Table(title="Confusion (predicted × generation-based ground truth)", show_lines=True)
    cm.add_column("")
    cm.add_column("actual: REFUSAL", justify="right")
    cm.add_column("actual: NOT-REFUSAL", justify="right")
    cm.add_row("predicted: REFUSAL", str(summary.true_positive), f"[red]{summary.false_positive}[/red]")
    cm.add_row("predicted: NOT-REFUSAL", f"[red]{summary.false_negative}[/red]", str(summary.true_negative))
    console.print(cm)

    tp, fp, tn, fn = (summary.true_positive, summary.false_positive,
                     summary.true_negative, summary.false_negative)
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    console.print(
        f"Accuracy: [bold]{acc:.3f}[/bold]  "
        f"Precision: {precision:.3f}  Recall: {recall:.3f}  "
        f"FPR (harmless mislabeled): [yellow]{fpr:.3f}[/yellow]  "
        f"FNR (missed refusal): [yellow]{fnr:.3f}[/yellow]"
    )

    # Rates by source
    rates = Table(title="Refusal rates by source", show_lines=False)
    rates.add_column("source"); rates.add_column("detector predicts", justify="right"); rates.add_column("model actually", justify="right")
    rates.add_row("harmful",
                  f"{summary.harmful_predicted_refusal_rate:.3f}",
                  f"{summary.harmful_generated_refusal_rate:.3f}")
    rates.add_row("harmless",
                  f"{summary.harmless_predicted_refusal_rate:.3f}",
                  f"{summary.harmless_generated_refusal_rate:.3f}")
    console.print(rates)

    # Score distribution
    dist = Table(title="Max-anchor log-prob distribution", show_lines=False)
    dist.add_column("source"); dist.add_column("min", justify="right")
    dist.add_column("p05", justify="right"); dist.add_column("median", justify="right")
    dist.add_column("p95", justify="right"); dist.add_column("max", justify="right")
    dist.add_column("mean", justify="right")
    for src in ("harmful", "harmless", "all"):
        s = summary.score_stats.get(src, {})
        if not s:
            continue
        dist.add_row(
            src,
            f"{s['min']:.2f}", f"{s['p05']:.2f}", f"{s['median']:.2f}",
            f"{s['p95']:.2f}", f"{s['max']:.2f}", f"{s['mean']:.2f}",
        )
    console.print(dist)

    # Which anchors cause false positives
    if summary.fp_by_anchor:
        console.print("\n[bold]False-positive anchors (harmless prompts labeled as refusal by this anchor):[/bold]")
        for a, n in sorted(summary.fp_by_anchor.items(), key=lambda x: -x[1]):
            console.print(f"  [red]{a!r}[/red]: {n} case(s)")

    # Suggested threshold
    if summary.suggested_threshold is not None:
        delta = summary.suggested_threshold - summary.threshold
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        console.print(
            f"\n[bold]Suggested threshold[/bold] (max accuracy on this sample): "
            f"[cyan]{summary.suggested_threshold:.2f}[/cyan]  {arrow}  "
            f"(current: {summary.threshold})  → accuracy: {summary.suggested_threshold_accuracy:.3f}"
        )

    # Example FPs / FNs
    fps = [a for a in audits if a.predicted_refusal and not a.generation_is_refusal]
    fns = [a for a in audits if not a.predicted_refusal and a.generation_is_refusal]
    if fps:
        console.rule("[bold red]False positives (detector said refuse, model actually complied)[/bold red]")
        for a in fps[:6]:
            _print_case(console, a)
    if fns:
        console.rule("[bold yellow]False negatives (detector said allow, model actually refused)[/bold yellow]")
        for a in fns[:6]:
            _print_case(console, a)


def _print_case(console, a: PromptAudit) -> None:
    console.print(
        f"[dim]#{a.idx} ({a.source}) score={a.max_score:.2f} via {a.winning_anchor!r}[/dim]"
    )
    prompt_preview = a.prompt[:120] + ("..." if len(a.prompt) > 120 else "")
    gen_preview = a.generated_text[:180].replace("\n", " ")
    if len(a.generated_text) > 180:
        gen_preview += "..."
    console.print(f"  prompt: {prompt_preview}")
    console.print(f"  gen:    {gen_preview}\n")
