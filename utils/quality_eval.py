#!/usr/bin/env python3
"""
Quality evaluation: KL divergence and perplexity comparison between base and abliterated models.

Loads two models sequentially (base first, then abliterated) to fit on a single GPU,
computes per-token perplexity and KL divergence on a shared set of reference prompts.
Uses top-k KL approximation (same pattern as src/kl_monitor.py).
"""

import gc
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model_utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality comparison between base and abliterated models."""

    base_model: str
    abliterated_model: str
    num_prompts: int
    base_perplexity: float
    abliterated_perplexity: float
    perplexity_delta: float
    perplexity_ratio: float
    mean_kl_divergence: float
    median_kl_divergence: float
    max_kl_divergence: float
    std_kl_divergence: float
    per_prompt: list[dict] = field(default_factory=list)
    eval_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "abliterated_model": self.abliterated_model,
            "num_prompts": self.num_prompts,
            "base_perplexity": self.base_perplexity,
            "abliterated_perplexity": self.abliterated_perplexity,
            "perplexity_delta": self.perplexity_delta,
            "perplexity_ratio": self.perplexity_ratio,
            "mean_kl_divergence": self.mean_kl_divergence,
            "median_kl_divergence": self.median_kl_divergence,
            "max_kl_divergence": self.max_kl_divergence,
            "std_kl_divergence": self.std_kl_divergence,
            "per_prompt": self.per_prompt,
            "eval_time_seconds": self.eval_time_seconds,
        }


def _format_prompt(tokenizer, prompt: str) -> str:
    """Format prompt with chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def _forward_batch(model, tokenizer, prompts: list[str]) -> list[dict]:
    """
    Run a single forward pass on a batch of prompts.

    Returns per-prompt dicts with:
        - logits: (seq_len, vocab_size) float32 tensor on CPU (padding stripped)
        - input_ids: (seq_len,) tensor on CPU (padding stripped)
    """
    formatted = [_format_prompt(tokenizer, p) for p in prompts]

    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            all_logits = outputs.logits  # (batch, seq_len, vocab)

        attention_mask = inputs.attention_mask
        input_ids = inputs.input_ids
        results = []

        for i in range(all_logits.size(0)):
            mask = attention_mask[i]
            non_pad = mask.nonzero(as_tuple=True)[0]
            start = non_pad[0].item() if len(non_pad) > 0 else 0

            results.append({
                "logits": all_logits[i, start:].cpu().float(),
                "input_ids": input_ids[i, start:].cpu(),
            })

        del all_logits, outputs, inputs
        return results
    finally:
        tokenizer.padding_side = original_padding_side


def _extract_perplexity(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute teacher-forced next-token perplexity from logits and input IDs."""
    if logits.size(0) <= 1:
        return float("nan")
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]
    ce_loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
    return ce_loss.exp().item()


def _extract_top_k(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k log probs and indices from logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.topk(top_k, dim=-1)  # (values, indices)


def _compute_kl(
    ref_log_probs: torch.Tensor,
    ref_indices: torch.Tensor,
    abl_full_log_probs: torch.Tensor,
) -> float:
    """
    Compute KL(P_base || P_abl) using base's top-k indices gathered from
    abliterated model's full log probs (same approach as kl_monitor.py).
    """
    min_len = min(ref_log_probs.size(0), abl_full_log_probs.size(0))
    ref_lp = ref_log_probs[:min_len]
    ref_idx = ref_indices[:min_len]
    abl_lp = abl_full_log_probs[:min_len]

    # Clamp indices to valid vocab range (guards against mismatched vocab sizes)
    vocab_size = abl_lp.size(-1)
    ref_idx = ref_idx.clamp(max=vocab_size - 1)

    abl_at_ref = abl_lp.gather(1, ref_idx)

    p_ref = ref_lp.exp()
    kl_per_position = (p_ref * (ref_lp - abl_at_ref)).sum(dim=-1)
    kl_per_position = kl_per_position.clamp(min=0.0)

    return kl_per_position.mean().item()


def compare_quality(
    base_model_path: str,
    abliterated_model_path: str,
    prompts: list[str],
    batch_size: int = 4,
    top_k: int = 200,
    device: str = "cuda",
    dtype: str = "float16",
    output_dir: Optional[str] = None,
) -> QualityResult:
    """
    Compare quality between base and abliterated models using perplexity and KL divergence.

    Loads models sequentially (base first, free, then abliterated) to fit in single GPU.
    Computes perplexity for both models and KL(P_base || P_abliterated) using top-k
    approximation.

    Args:
        base_model_path: Path to the base (original) model
        abliterated_model_path: Path to the abliterated model
        prompts: Reference prompts to evaluate on
        batch_size: Batch size for forward passes
        top_k: Number of top tokens for KL approximation
        device: Device to use
        dtype: Model precision ("float16", "bfloat16", "float32")
        output_dir: Directory to save JSON report (optional)

    Returns:
        QualityResult with perplexity and KL divergence metrics
    """
    if not prompts:
        raise ValueError("prompts list cannot be empty")

    start_time = time.perf_counter()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    num_prompts = len(prompts)
    logger.info(f"Quality eval: {num_prompts} prompts, batch_size={batch_size}, top_k={top_k}")

    # =========================================================================
    # Phase 1: Load base model, compute top-k log probs + perplexity, cache
    # =========================================================================
    logger.info(f"Loading base model: {base_model_path}")
    base_model, base_tokenizer = load_model_and_tokenizer(
        base_model_path, device=device, dtype=torch_dtype, trust_remote_code=True
    )

    # Cached per-prompt: (top_k_log_probs, top_k_indices)
    base_cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    base_perplexities: list[float] = []

    logger.info("Computing base model logits...")
    for i in tqdm(range(0, num_prompts, batch_size), desc="Base model", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        batch_results = _forward_batch(base_model, base_tokenizer, batch_prompts)

        for r in batch_results:
            base_perplexities.append(_extract_perplexity(r["logits"], r["input_ids"]))
            top_values, top_indices = _extract_top_k(r["logits"], top_k)
            base_cached.append((top_values, top_indices))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free base model
    logger.info("Freeing base model from GPU...")
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Load abliterated model, compute perplexity + KL in one pass
    # =========================================================================
    logger.info(f"Loading abliterated model: {abliterated_model_path}")
    abl_model, abl_tokenizer = load_model_and_tokenizer(
        abliterated_model_path, device=device, dtype=torch_dtype, trust_remote_code=True
    )

    abl_perplexities: list[float] = []
    per_prompt_kl: list[float] = []

    logger.info("Computing abliterated model logits and KL divergence...")
    prompt_idx = 0
    for i in tqdm(range(0, num_prompts, batch_size), desc="Abliterated model", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        batch_results = _forward_batch(abl_model, abl_tokenizer, batch_prompts)

        for r in batch_results:
            # Perplexity
            abl_perplexities.append(_extract_perplexity(r["logits"], r["input_ids"]))

            # KL divergence: use full log probs from abl, gather at base's top-k indices
            if prompt_idx < len(base_cached):
                ref_lp, ref_idx = base_cached[prompt_idx]
                abl_full_lp = F.log_softmax(r["logits"], dim=-1)
                kl = _compute_kl(ref_lp, ref_idx, abl_full_lp)
                per_prompt_kl.append(kl)
                del abl_full_lp
            prompt_idx += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free abliterated model
    del abl_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 3: Aggregate results
    # =========================================================================
    valid_base_ppl = [p for p in base_perplexities if not math.isnan(p)]
    valid_abl_ppl = [p for p in abl_perplexities if not math.isnan(p)]

    base_ppl = sum(valid_base_ppl) / len(valid_base_ppl) if valid_base_ppl else float("nan")
    abl_ppl = sum(valid_abl_ppl) / len(valid_abl_ppl) if valid_abl_ppl else float("nan")

    if not per_prompt_kl:
        logger.warning("No valid KL divergence values computed. All prompts may be too short.")
        kl_tensor = torch.tensor([float("nan")])
    else:
        kl_tensor = torch.tensor(per_prompt_kl)

    per_prompt_results = []
    for idx in range(num_prompts):
        per_prompt_results.append({
            "prompt": prompts[idx][:200],
            "base_ppl": base_perplexities[idx] if idx < len(base_perplexities) else None,
            "abl_ppl": abl_perplexities[idx] if idx < len(abl_perplexities) else None,
            "kl_div": per_prompt_kl[idx] if idx < len(per_prompt_kl) else None,
        })

    elapsed = time.perf_counter() - start_time

    result = QualityResult(
        base_model=base_model_path,
        abliterated_model=abliterated_model_path,
        num_prompts=num_prompts,
        base_perplexity=base_ppl,
        abliterated_perplexity=abl_ppl,
        perplexity_delta=abl_ppl - base_ppl,
        perplexity_ratio=abl_ppl / base_ppl if base_ppl > 0 else float("nan"),
        mean_kl_divergence=kl_tensor.mean().item(),
        median_kl_divergence=kl_tensor.median().item(),
        max_kl_divergence=kl_tensor.max().item(),
        std_kl_divergence=kl_tensor.std().item() if len(kl_tensor) > 1 else 0.0,
        per_prompt=per_prompt_results,
        eval_time_seconds=elapsed,
    )

    if output_dir:
        _save_report(result, output_dir)

    return result


def _save_report(result: QualityResult, output_dir: str) -> Path:
    """Save quality evaluation report as JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(result.base_model).name.replace("/", "_").replace("\\", "_")
    abl_name = Path(result.abliterated_model).name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_eval_{base_name}_vs_{abl_name}_{timestamp}.json"

    report_path = output_path / filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Quality report saved to {report_path}")
    return report_path


def load_prompts_for_quality_eval(
    path: Optional[str] = None,
    num_prompts: int = 50,
) -> list[str]:
    """
    Load reference prompts for quality evaluation.

    Uses harmless.txt by default (general capability prompts).

    Args:
        path: Path to prompts file (one per line). None = use default harmless.txt
        num_prompts: Maximum number of prompts to use

    Returns:
        List of prompt strings
    """
    import random
    from src.abliterate import load_prompts_from_file, get_default_prompts_path

    if path is not None:
        prompts = load_prompts_from_file(path, num_prompts=None)
    else:
        harmless_path = get_default_prompts_path("harmless.txt")
        prompts = load_prompts_from_file(harmless_path, num_prompts=None)

    if len(prompts) > num_prompts:
        random.seed(42)
        prompts = random.sample(prompts, num_prompts)

    return prompts
