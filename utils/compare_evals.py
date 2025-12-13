#!/usr/bin/env python3
"""Compare two lm_eval JSON result files and display differences."""

import argparse
import json
import sys
from pathlib import Path


def load_results(filepath: str) -> dict:
    """Load JSON results file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict[str, dict[str, float]]:
    """Extract benchmark metrics from results."""
    metrics = {}
    results = data.get("results", {})

    for task_name, task_data in results.items():
        metrics[task_name] = {}
        for key, value in task_data.items():
            if key == "alias":
                continue
            if isinstance(value, (int, float)):
                # Clean up metric name (remove ",none" suffix)
                metric_name = key.replace(",none", "")
                metrics[task_name][metric_name] = value

    return metrics


def is_refusal_eval(data: dict) -> bool:
    """Check if this is a refusal rate evaluation result."""
    results = data.get("results", {})
    return "harmful_prompts" in results or "harmless_prompts" in results


def compare_results(file1: str, file2: str, show_all: bool = False) -> None:
    """Compare two result files and print differences."""
    data1 = load_results(file1)
    data2 = load_results(file2)

    metrics1 = extract_metrics(data1)
    metrics2 = extract_metrics(data2)

    # Get model info if available
    name1 = Path(file1).stem
    name2 = Path(file2).stem

    print(f"\n{'='*70}")
    print("EVALUATION COMPARISON")
    print(f"{'='*70}")
    print(f"File 1: {name1}")
    print(f"File 2: {name2}")
    print(f"{'='*70}\n")

    # Get all tasks from both files
    all_tasks = sorted(set(metrics1.keys()) | set(metrics2.keys()))

    # Print header
    print(f"{'Task':<15} {'Metric':<12} {'File 1':>10} {'File 2':>10} {'Delta':>10} {'%':>8}")
    print("-" * 70)

    total_delta = 0.0
    num_metrics = 0

    for task in all_tasks:
        task_metrics1 = metrics1.get(task, {})
        task_metrics2 = metrics2.get(task, {})

        all_metrics = sorted(set(task_metrics1.keys()) | set(task_metrics2.keys()))
        # Filter to main accuracy metrics unless show_all
        if not show_all:
            all_metrics = [m for m in all_metrics if not m.endswith("_stderr")]

        for metric in all_metrics:
            val1 = task_metrics1.get(metric)
            val2 = task_metrics2.get(metric)

            if val1 is None:
                val1_str = "N/A"
                delta_str = ""
                pct_str = ""
            elif val2 is None:
                val1_str = f"{val1:.3f}"
                delta_str = ""
                pct_str = ""
            else:
                val1_str = f"{val1:.3f}"
                val2_str = f"{val2:.3f}"
                delta = val2 - val1
                pct = (delta / val1 * 100) if val1 != 0 else 0

                # Color coding via symbols
                if delta > 0.001:
                    delta_str = f"+{delta:.3f}"
                    pct_str = f"+{pct:.1f}%"
                elif delta < -0.001:
                    delta_str = f"{delta:.3f}"
                    pct_str = f"{pct:.1f}%"
                else:
                    delta_str = f"{delta:.3f}"
                    pct_str = "0.0%"

                if not metric.endswith("_stderr"):
                    total_delta += delta
                    num_metrics += 1

                print(f"{task:<15} {metric:<12} {val1_str:>10} {val2_str:>10} {delta_str:>10} {pct_str:>8}")
                continue

            val2_str = f"{val2:.3f}" if val2 is not None else "N/A"
            print(f"{task:<15} {metric:<12} {val1_str:>10} {val2_str:>10} {delta_str:>10} {pct_str:>8}")

    print("-" * 70)
    if num_metrics > 0:
        avg_delta = total_delta / num_metrics
        print(f"\nAverage delta across {num_metrics} metrics: {avg_delta:+.4f} ({avg_delta*100:+.2f}%)")

    # Special interpretation for refusal evals
    if is_refusal_eval(data1) or is_refusal_eval(data2):
        print("\n" + "-" * 70)
        print("REFUSAL EVAL INTERPRETATION")
        print("-" * 70)
        print("For ABLITERATED models (vs base):")
        print("  - harmful_prompts refusal_rate ↓ = abliteration working (less refusals)")
        print("  - harmless_prompts refusal_rate ~ = no over-refusal introduced")

    # Print metadata comparison
    print(f"\n{'='*70}")
    print("METADATA")
    print(f"{'='*70}")

    config1 = data1.get("config", {})
    config2 = data2.get("config", {})

    print(f"{'Property':<25} {'File 1':<30} {'File 2':<30}")
    print("-" * 70)

    for key in ["model_num_parameters", "model_dtype", "batch_size", "limit"]:
        v1 = config1.get(key, "N/A")
        v2 = config2.get(key, "N/A")
        if isinstance(v1, float):
            v1 = f"{v1:.0f}"
        if isinstance(v2, float):
            v2 = f"{v2:.0f}"
        print(f"{key:<25} {str(v1):<30} {str(v2):<30}")

    # Evaluation time
    time1 = data1.get("total_evaluation_time_seconds", "N/A")
    time2 = data2.get("total_evaluation_time_seconds", "N/A")
    if time1 != "N/A":
        time1 = f"{float(time1):.1f}s"
    if time2 != "N/A":
        time2 = f"{float(time2):.1f}s"
    print(f"{'eval_time':<25} {time1:<30} {time2:<30}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two lm_eval JSON result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_evals.py results1.json results2.json
  python compare_evals.py --all results1.json results2.json  # Include stderr
        """
    )
    parser.add_argument("file1", help="First results JSON file")
    parser.add_argument("file2", help="Second results JSON file")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all metrics including stderr")

    args = parser.parse_args()

    # Validate files exist
    for f in [args.file1, args.file2]:
        if not Path(f).exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    compare_results(args.file1, args.file2, show_all=args.all)


if __name__ == "__main__":
    main()
