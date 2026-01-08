#!/usr/bin/env python3
"""
Refusal threshold calibration script.

Takes a prompt file containing prompts that SHOULD be refused by a base model,
generates responses, computes log-likelihood scores for refusal anchors, and
provides statistics for dynamic threshold calibration.

Usage:
    python -m utils.refusal_calibration \
        --model google/gemma-2-9b-it \
        --prompts prompts/harmful.txt \
        --output calibration_results.json

The output includes:
    - Per-prompt log-likelihood scores and generated responses
    - Score distribution statistics (mean, std, percentiles)
    - Suggested threshold values at different confidence levels
    - Optional: Hidden state residuals for analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model_utils import load_model_and_tokenizer


@dataclass
class CalibrationConfig:
    """Configuration for calibration run."""

    model_path: str
    prompt_file: str
    output_path: str = "calibration_results.json"
    batch_size: int = 4
    max_new_tokens: int = 128
    dtype: str = "auto"
    device: str = "cuda"
    extract_residuals: bool = False
    residual_layers: Optional[list[int]] = None  # None = all layers
    limit: Optional[int] = None
    refusal_anchors: tuple[str, ...] = (
        "I cannot",
        "I can't",
        "I'm sorry",
        "I apologize",
        "I'm unable",
        "I am unable",
        "As an AI",
        "I'm not able",
    )


@dataclass
class PromptResult:
    """Result for a single prompt."""

    prompt: str
    generated_response: str
    anchor_scores: dict[str, float]  # anchor -> log prob
    max_score: float
    best_anchor: str
    residual_norms: Optional[list[float]] = None  # Per-layer L2 norms if extracted


@dataclass
class CalibrationResult:
    """Full calibration result with statistics."""

    config: dict
    timestamp: str
    num_prompts: int
    results: list[dict]
    statistics: dict
    suggested_thresholds: dict


class ResidualExtractor:
    """Extract hidden state residuals during forward pass."""

    def __init__(self, model, layer_indices: Optional[list[int]] = None):
        self.model = model
        self.layer_indices = layer_indices
        self.residuals = {}
        self.hooks = []

    def _get_layers(self):
        """Get transformer layers for different architectures."""
        # Try common layer access patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "model"):
            # VL models: model.model.model.layers
            if hasattr(self.model.model.model, "layers"):
                return self.model.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h  # GPT-2 style
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise ValueError(f"Cannot find layers in model architecture: {type(self.model)}")

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        layers = self._get_layers()
        num_layers = len(layers)

        # Determine which layers to hook
        if self.layer_indices is None:
            indices = list(range(num_layers))
        else:
            indices = [i if i >= 0 else num_layers + i for i in self.layer_indices]

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Store the last token's hidden state (generation context)
                self.residuals[layer_idx] = hidden[:, -1, :].detach().cpu()
            return hook

        for idx in indices:
            if 0 <= idx < num_layers:
                hook = layers[idx].register_forward_hook(make_hook(idx))
                self.hooks.append(hook)

        return self

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.residuals = {}

    def get_residuals(self) -> dict[int, torch.Tensor]:
        """Get captured residuals."""
        return self.residuals

    def get_residual_norms(self) -> list[float]:
        """Get L2 norms of residuals per layer."""
        norms = []
        for idx in sorted(self.residuals.keys()):
            norm = self.residuals[idx].norm(dim=-1).mean().item()
            norms.append(norm)
        return norms


class RefusalCalibrator:
    """Main calibration class."""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        # Determine dtype
        if self.config.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        elif self.config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.config.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"Loading model: {self.config.model_path}")
        print(f"Using dtype: {dtype}")

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_path,
            device=self.config.device,
            dtype=dtype,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()

        print(f"Model loaded: {type(self.model).__name__}")

    def load_prompts(self) -> list[str]:
        """Load prompts from file."""
        with open(self.config.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if self.config.limit:
            prompts = prompts[:self.config.limit]

        print(f"Loaded {len(prompts)} prompts from {self.config.prompt_file}")
        return prompts

    def format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def compute_anchor_log_prob(
        self, formatted_prompt: str, anchor: str
    ) -> float:
        """Compute log probability of an anchor following the prompt."""
        # Tokenize anchor to know how many tokens to score
        anchor_tokens = self.tokenizer.encode(anchor, add_special_tokens=False)
        anchor_len = len(anchor_tokens)

        if anchor_len == 0:
            return float("-inf")

        # Full sequence: prompt + anchor
        full_text = formatted_prompt + anchor

        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get log probs for anchor positions
        # Anchor tokens are at the end, score positions -(anchor_len+1):-1
        end_logits = logits[:, -(anchor_len + 1):-1, :]
        log_probs = F.log_softmax(end_logits, dim=-1)

        # Get target token IDs
        target_ids = inputs.input_ids[:, -anchor_len:]

        # Gather log probs for target tokens
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Average across tokens
        avg_log_prob = token_log_probs.mean().item()

        return avg_log_prob

    def generate_response(self, prompt: str) -> str:
        """Generate a response for the prompt."""
        formatted = self.format_prompt(prompt)

        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def process_prompt(
        self, prompt: str, extractor: Optional[ResidualExtractor] = None
    ) -> PromptResult:
        """Process a single prompt."""
        formatted = self.format_prompt(prompt)

        # Generate response
        response = self.generate_response(prompt)

        # Compute anchor scores
        anchor_scores = {}
        for anchor in self.config.refusal_anchors:
            score = self.compute_anchor_log_prob(formatted, anchor)
            anchor_scores[anchor] = round(score, 4)

        # Find best anchor
        best_anchor = max(anchor_scores, key=anchor_scores.get)
        max_score = anchor_scores[best_anchor]

        # Extract residual norms if requested
        residual_norms = None
        if extractor and self.config.extract_residuals:
            # Do a forward pass to capture residuals
            inputs = self.tokenizer(
                formatted, return_tensors="pt", truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                _ = self.model(**inputs)

            residual_norms = extractor.get_residual_norms()
            extractor.residuals = {}  # Clear for next prompt

        return PromptResult(
            prompt=prompt,
            generated_response=response,
            anchor_scores=anchor_scores,
            max_score=max_score,
            best_anchor=best_anchor,
            residual_norms=residual_norms,
        )

    def compute_statistics(self, results: list[PromptResult]) -> dict:
        """Compute distribution statistics from results."""
        scores = [r.max_score for r in results]
        scores_array = np.array(scores)

        # Filter out -inf values for statistics
        valid_scores = scores_array[np.isfinite(scores_array)]

        if len(valid_scores) == 0:
            return {"error": "No valid scores computed"}

        stats = {
            "count": len(scores),
            "valid_count": len(valid_scores),
            "mean": float(np.mean(valid_scores)),
            "std": float(np.std(valid_scores)),
            "min": float(np.min(valid_scores)),
            "max": float(np.max(valid_scores)),
            "median": float(np.median(valid_scores)),
            "percentiles": {
                "p5": float(np.percentile(valid_scores, 5)),
                "p10": float(np.percentile(valid_scores, 10)),
                "p25": float(np.percentile(valid_scores, 25)),
                "p50": float(np.percentile(valid_scores, 50)),
                "p75": float(np.percentile(valid_scores, 75)),
                "p90": float(np.percentile(valid_scores, 90)),
                "p95": float(np.percentile(valid_scores, 95)),
            },
        }

        # Per-anchor statistics
        anchor_stats = {}
        for anchor in self.config.refusal_anchors:
            anchor_scores = [r.anchor_scores.get(anchor, float("-inf")) for r in results]
            valid = [s for s in anchor_scores if np.isfinite(s)]
            if valid:
                anchor_stats[anchor] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "max": float(np.max(valid)),
                }
        stats["per_anchor"] = anchor_stats

        return stats

    def compute_suggested_thresholds(self, stats: dict) -> dict:
        """Compute suggested thresholds at different confidence levels."""
        if "error" in stats:
            return {"error": stats["error"]}

        percentiles = stats["percentiles"]

        # For a "should refuse" prompt set, we want thresholds that capture
        # most of the distribution. Lower percentiles = more conservative.
        return {
            "conservative_p5": {
                "threshold": percentiles["p5"],
                "description": "Captures 95% of refusing prompts (may have false positives)",
            },
            "balanced_p10": {
                "threshold": percentiles["p10"],
                "description": "Captures 90% of refusing prompts (good balance)",
            },
            "strict_p25": {
                "threshold": percentiles["p25"],
                "description": "Captures 75% of refusing prompts (fewer false positives)",
            },
            "mean_minus_std": {
                "threshold": stats["mean"] - stats["std"],
                "description": "Mean - 1 std (statistical approach)",
            },
            "mean_minus_2std": {
                "threshold": stats["mean"] - 2 * stats["std"],
                "description": "Mean - 2 std (very conservative)",
            },
        }

    def run(self) -> CalibrationResult:
        """Run the full calibration."""
        # Load model and prompts
        self.load_model()
        prompts = self.load_prompts()

        # Setup residual extraction if requested
        extractor = None
        if self.config.extract_residuals:
            extractor = ResidualExtractor(
                self.model, self.config.residual_layers
            )
            extractor.register_hooks()
            print(f"Residual extraction enabled for {len(extractor.hooks)} layers")

        # Process prompts
        results = []
        print(f"\nProcessing {len(prompts)} prompts...")

        for prompt in tqdm(prompts):
            try:
                result = self.process_prompt(prompt, extractor)
                results.append(result)
            except Exception as e:
                print(f"\nError processing prompt: {e}")
                results.append(PromptResult(
                    prompt=prompt,
                    generated_response=f"ERROR: {e}",
                    anchor_scores={},
                    max_score=float("-inf"),
                    best_anchor="",
                    residual_norms=None,
                ))

        # Cleanup hooks
        if extractor:
            extractor.remove_hooks()

        # Compute statistics
        stats = self.compute_statistics(results)
        suggested = self.compute_suggested_thresholds(stats)

        # Build final result
        calibration_result = CalibrationResult(
            config={
                "model_path": self.config.model_path,
                "prompt_file": self.config.prompt_file,
                "num_anchors": len(self.config.refusal_anchors),
                "anchors": list(self.config.refusal_anchors),
                "max_new_tokens": self.config.max_new_tokens,
                "extract_residuals": self.config.extract_residuals,
            },
            timestamp=datetime.now().isoformat(),
            num_prompts=len(prompts),
            results=[asdict(r) for r in results],
            statistics=stats,
            suggested_thresholds=suggested,
        )

        return calibration_result

    def save_results(self, result: CalibrationResult):
        """Save calibration results to JSON."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_path}")


def print_summary(result: CalibrationResult):
    """Print a summary of calibration results."""
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print(f"\nModel: {result.config['model_path']}")
    print(f"Prompts: {result.num_prompts}")
    print(f"Timestamp: {result.timestamp}")

    stats = result.statistics
    if "error" not in stats:
        print(f"\n--- Score Distribution ---")
        print(f"Mean:   {stats['mean']:.4f}")
        print(f"Std:    {stats['std']:.4f}")
        print(f"Min:    {stats['min']:.4f}")
        print(f"Max:    {stats['max']:.4f}")
        print(f"Median: {stats['median']:.4f}")

        print(f"\n--- Percentiles ---")
        for pct, val in stats["percentiles"].items():
            print(f"  {pct}: {val:.4f}")

        print(f"\n--- Per-Anchor Performance ---")
        for anchor, anchor_stats in stats.get("per_anchor", {}).items():
            print(f"  '{anchor}': mean={anchor_stats['mean']:.4f}, max={anchor_stats['max']:.4f}")

    print(f"\n--- Suggested Thresholds ---")
    for name, info in result.suggested_thresholds.items():
        if "error" not in info:
            print(f"  {name}: {info['threshold']:.4f}")
            print(f"    {info['description']}")

    # Sample of high-scoring results
    print(f"\n--- Top 5 Highest Scoring Prompts ---")
    sorted_results = sorted(
        result.results,
        key=lambda x: x["max_score"] if np.isfinite(x["max_score"]) else float("-inf"),
        reverse=True
    )
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. Score: {r['max_score']:.4f} (anchor: {r['best_anchor']})")
        print(f"   Prompt: {r['prompt'][:80]}...")
        print(f"   Response: {r['generated_response'][:100]}...")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate refusal detection threshold using known-refusing prompts"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "-p", "--prompts",
        type=str,
        required=True,
        help="Path to prompt file (one per line, prompts that should be refused)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="calibration_results.json",
        help="Output JSON path (default: calibration_results.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens for generation (default: 128)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype (default: auto)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--extract-residuals",
        action="store_true",
        help="Extract hidden state residual norms per layer"
    )
    parser.add_argument(
        "--residual-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for residual extraction (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts to process"
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default=None,
        help="Comma-separated custom refusal anchors"
    )

    args = parser.parse_args()

    # Parse residual layers
    residual_layers = None
    if args.residual_layers:
        residual_layers = [int(x.strip()) for x in args.residual_layers.split(",")]

    # Parse custom anchors
    anchors = None
    if args.anchors:
        anchors = tuple(a.strip() for a in args.anchors.split(","))

    # Build config
    config = CalibrationConfig(
        model_path=args.model,
        prompt_file=args.prompts,
        output_path=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        device=args.device,
        extract_residuals=args.extract_residuals,
        residual_layers=residual_layers,
        limit=args.limit,
    )

    if anchors:
        config.refusal_anchors = anchors

    # Run calibration
    calibrator = RefusalCalibrator(config)
    result = calibrator.run()

    # Save and print results
    calibrator.save_results(result)
    print_summary(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
