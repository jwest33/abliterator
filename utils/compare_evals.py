#!/usr/bin/env python3
"""Compare two or three lm_eval JSON result files and display differences."""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricComparison:
    """Container for a single metric comparison."""
    task: str
    metric: str
    val1: float
    val2: float
    delta: float
    pct_change: float  # Percentage change: (val2 - val1) / val1 * 100


@dataclass
class AggregateStats:
    """Container for aggregate statistics across all metrics."""
    comparisons: list[MetricComparison] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.comparisons)

    @property
    def improvements(self) -> list[MetricComparison]:
        """Metrics where File 2 > File 1 (higher is better assumed)."""
        return [c for c in self.comparisons if c.delta > 0.001]

    @property
    def regressions(self) -> list[MetricComparison]:
        """Metrics where File 2 < File 1."""
        return [c for c in self.comparisons if c.delta < -0.001]

    @property
    def unchanged(self) -> list[MetricComparison]:
        """Metrics with negligible change."""
        return [c for c in self.comparisons if abs(c.delta) <= 0.001]

    def mean_pct_change(self) -> float:
        """Average percentage change across all metrics."""
        if not self.comparisons:
            return 0.0
        # Filter out infinite/nan values (from division by zero)
        valid = [c.pct_change for c in self.comparisons if math.isfinite(c.pct_change)]
        return sum(valid) / len(valid) if valid else 0.0

    def median_pct_change(self) -> float:
        """Median percentage change (more robust to outliers)."""
        valid = sorted([c.pct_change for c in self.comparisons if math.isfinite(c.pct_change)])
        if not valid:
            return 0.0
        n = len(valid)
        if n % 2 == 0:
            return (valid[n // 2 - 1] + valid[n // 2]) / 2
        return valid[n // 2]

    def std_pct_change(self) -> float:
        """Standard deviation of percentage changes."""
        valid = [c.pct_change for c in self.comparisons if math.isfinite(c.pct_change)]
        if len(valid) < 2:
            return 0.0
        mean = sum(valid) / len(valid)
        variance = sum((x - mean) ** 2 for x in valid) / (len(valid) - 1)
        return math.sqrt(variance)

    def mean_delta(self) -> float:
        """Average raw delta (only meaningful for same-scale metrics)."""
        if not self.comparisons:
            return 0.0
        return sum(c.delta for c in self.comparisons) / len(self.comparisons)

    def geometric_mean_ratio(self) -> float:
        """Geometric mean of val2/val1 ratios (useful for multiplicative comparisons)."""
        ratios = []
        for c in self.comparisons:
            if c.val1 > 0 and c.val2 > 0:
                ratios.append(c.val2 / c.val1)
        if not ratios:
            return 1.0
        log_sum = sum(math.log(r) for r in ratios)
        return math.exp(log_sum / len(ratios))

    def min_pct_change(self) -> tuple[float, MetricComparison | None]:
        """Worst regression (most negative % change)."""
        valid = [(c.pct_change, c) for c in self.comparisons if math.isfinite(c.pct_change)]
        if not valid:
            return 0.0, None
        return min(valid, key=lambda x: x[0])

    def max_pct_change(self) -> tuple[float, MetricComparison | None]:
        """Best improvement (most positive % change)."""
        valid = [(c.pct_change, c) for c in self.comparisons if math.isfinite(c.pct_change)]
        if not valid:
            return 0.0, None
        return max(valid, key=lambda x: x[0])

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


def format_delta(delta: float, pct: float) -> tuple[str, str]:
    """Format delta and percentage for display."""
    if delta > 0.001:
        delta_str = f"+{delta:.3f}"
        pct_str = f"+{pct:.1f}%" if math.isfinite(pct) else "+inf%"
    elif delta < -0.001:
        delta_str = f"{delta:.3f}"
        pct_str = f"{pct:.1f}%" if math.isfinite(pct) else "-inf%"
    else:
        delta_str = f"{delta:.3f}"
        pct_str = "0.0%"
    return delta_str, pct_str


def compute_pct_change(val1: float, delta: float) -> float:
    """Compute percentage change, handling division by zero."""
    if val1 != 0:
        return (delta / val1) * 100
    elif delta > 0:
        return float('inf')
    elif delta < 0:
        return float('-inf')
    return 0.0


def print_aggregate_stats(stats: AggregateStats, label: str, line_width: int) -> None:
    """Print aggregate statistics for a comparison."""
    if stats.count == 0:
        return

    print(f"\n--- {label} ---")

    # Summary counts
    n_improved = len(stats.improvements)
    n_regressed = len(stats.regressions)
    n_unchanged = len(stats.unchanged)
    print(f"Metrics compared: {stats.count}")
    print(f"  Improved:  {n_improved:3d} ({100*n_improved/stats.count:.1f}%)")
    print(f"  Regressed: {n_regressed:3d} ({100*n_regressed/stats.count:.1f}%)")
    print(f"  Unchanged: {n_unchanged:3d} ({100*n_unchanged/stats.count:.1f}%)")

    # Central tendency measures
    print(f"Percentage Change:")
    print(f"  Mean:   {stats.mean_pct_change():+.2f}%")
    print(f"  Median: {stats.median_pct_change():+.2f}%")
    print(f"  Std:    {stats.std_pct_change():.2f}%")

    # Geometric mean ratio
    geo_mean = stats.geometric_mean_ratio()
    print(f"Geometric Mean Ratio: {geo_mean:.4f}")
    if geo_mean > 1:
        print(f"  (~{(geo_mean-1)*100:.2f}% better)")
    elif geo_mean < 1:
        print(f"  (~{(1-geo_mean)*100:.2f}% worse)")

    # Extremes
    min_pct, min_comp = stats.min_pct_change()
    max_pct, max_comp = stats.max_pct_change()
    if min_comp:
        print(f"Largest Regression: {min_comp.task}/{min_comp.metric}")
        print(f"  {min_comp.val1:.3f} -> {min_comp.val2:.3f} ({min_pct:+.2f}%)")
    if max_comp:
        print(f"Largest Improvement: {max_comp.task}/{max_comp.metric}")
        print(f"  {max_comp.val1:.3f} -> {max_comp.val2:.3f} ({max_pct:+.2f}%)")


def compare_results(file1: str, file2: str, file3: str | None = None, show_all: bool = False) -> None:
    """Compare two or three result files and print differences."""
    data1 = load_results(file1)
    data2 = load_results(file2)
    data3 = load_results(file3) if file3 else None

    metrics1 = extract_metrics(data1)
    metrics2 = extract_metrics(data2)
    metrics3 = extract_metrics(data3) if data3 else None

    # Get model info if available
    name1 = Path(file1).stem
    name2 = Path(file2).stem
    name3 = Path(file3).stem if file3 else None

    # Get all tasks from all files
    all_tasks = set(metrics1.keys()) | set(metrics2.keys())
    if metrics3:
        all_tasks |= set(metrics3.keys())
    all_tasks = sorted(all_tasks)

    # Calculate column widths dynamically
    task_width = max(len("Task"), max((len(t) for t in all_tasks), default=4))

    # Get all metric names to calculate metric column width
    all_metric_names = set()
    for task in all_tasks:
        all_metric_names.update(metrics1.get(task, {}).keys())
        all_metric_names.update(metrics2.get(task, {}).keys())
        if metrics3:
            all_metric_names.update(metrics3.get(task, {}).keys())
    if not show_all:
        all_metric_names = {m for m in all_metric_names if not m.endswith("_stderr")}
    metric_width = max(len("Metric"), max((len(m) for m in all_metric_names), default=6))

    # Calculate total line width based on number of files
    if file3:
        # Task + Metric + Base + File2 + Δ2 + %2 + File3 + Δ3 + %3
        line_width = max(120, task_width + metric_width + 8 + 8 + 8 + 7 + 8 + 8 + 7 + 10)
    else:
        line_width = max(70, task_width + metric_width + 10 + 10 + 10 + 8 + 5)

    print(f"\n{'='*line_width}")
    print("EVALUATION COMPARISON")
    print(f"{'='*line_width}")
    print(f"Base:   {name1}")
    print(f"File 2: {name2}")
    if name3:
        print(f"File 3: {name3}")
    print(f"{'='*line_width}\n")

    # Print header
    if file3:
        print(f"{'Task':<{task_width}} {'Metric':<{metric_width}} {'Base':>8} {'File2':>8} {'Δ2':>8} {'%2':>7} {'File3':>8} {'Δ3':>8} {'%3':>7}")
    else:
        print(f"{'Task':<{task_width}} {'Metric':<{metric_width}} {'File 1':>10} {'File 2':>10} {'Delta':>10} {'%':>8}")
    print("-" * line_width)

    stats2 = AggregateStats()
    stats3 = AggregateStats()

    for task in all_tasks:
        task_metrics1 = metrics1.get(task, {})
        task_metrics2 = metrics2.get(task, {})
        task_metrics3 = metrics3.get(task, {}) if metrics3 else {}

        all_metrics = set(task_metrics1.keys()) | set(task_metrics2.keys())
        if metrics3:
            all_metrics |= set(task_metrics3.keys())
        all_metrics = sorted(all_metrics)

        if not show_all:
            all_metrics = [m for m in all_metrics if not m.endswith("_stderr")]

        for metric in all_metrics:
            val1 = task_metrics1.get(metric)
            val2 = task_metrics2.get(metric)
            val3 = task_metrics3.get(metric) if metrics3 else None

            val1_str = f"{val1:.3f}" if val1 is not None else "N/A"

            # File 2 comparison
            if val1 is not None and val2 is not None:
                val2_str = f"{val2:.3f}"
                delta2 = val2 - val1
                pct2 = compute_pct_change(val1, delta2)
                delta2_str, pct2_str = format_delta(delta2, pct2)

                if not metric.endswith("_stderr"):
                    stats2.comparisons.append(MetricComparison(
                        task=task, metric=metric, val1=val1, val2=val2,
                        delta=delta2, pct_change=pct2,
                    ))
            else:
                val2_str = f"{val2:.3f}" if val2 is not None else "N/A"
                delta2_str, pct2_str = "", ""

            # File 3 comparison (if provided)
            if file3:
                if val1 is not None and val3 is not None:
                    val3_str = f"{val3:.3f}"
                    delta3 = val3 - val1
                    pct3 = compute_pct_change(val1, delta3)
                    delta3_str, pct3_str = format_delta(delta3, pct3)

                    if not metric.endswith("_stderr"):
                        stats3.comparisons.append(MetricComparison(
                            task=task, metric=metric, val1=val1, val2=val3,
                            delta=delta3, pct_change=pct3,
                        ))
                else:
                    val3_str = f"{val3:.3f}" if val3 is not None else "N/A"
                    delta3_str, pct3_str = "", ""

                print(f"{task:<{task_width}} {metric:<{metric_width}} {val1_str:>8} {val2_str:>8} {delta2_str:>8} {pct2_str:>7} {val3_str:>8} {delta3_str:>8} {pct3_str:>7}")
            else:
                print(f"{task:<{task_width}} {metric:<{metric_width}} {val1_str:>10} {val2_str:>10} {delta2_str:>10} {pct2_str:>8}")

    print("-" * line_width)

    # Print aggregate statistics
    if stats2.count > 0 or stats3.count > 0:
        print(f"\n{'='*line_width}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*line_width}")

        if stats2.count > 0:
            print_aggregate_stats(stats2, f"File2 vs Base", line_width)

        if stats3.count > 0:
            print_aggregate_stats(stats3, f"File3 vs Base", line_width)

    # Special interpretation for refusal evals
    any_refusal = is_refusal_eval(data1) or is_refusal_eval(data2) or (data3 and is_refusal_eval(data3))
    if any_refusal:
        print("\n" + "-" * line_width)
        print("REFUSAL EVAL INTERPRETATION")
        print("-" * line_width)
        print("For ABLITERATED models (vs base):")
        print("  - harmful_prompts refusal_rate ↓ = abliteration working (less refusals)")
        print("  - harmless_prompts refusal_rate ~ = no over-refusal introduced")

    # Print metadata comparison
    print(f"\n{'='*line_width}")
    print("METADATA")
    print(f"{'='*line_width}")

    config1 = data1.get("config", {})
    config2 = data2.get("config", {})
    config3 = data3.get("config", {}) if data3 else {}

    if file3:
        print(f"{'Property':<25} {'Base':<20} {'File 2':<20} {'File 3':<20}")
    else:
        print(f"{'Property':<25} {'File 1':<30} {'File 2':<30}")
    print("-" * line_width)

    for key in ["model_num_parameters", "model_dtype", "batch_size", "limit"]:
        v1 = config1.get(key, "N/A")
        v2 = config2.get(key, "N/A")
        v3 = config3.get(key, "N/A") if file3 else None
        if isinstance(v1, float):
            v1 = f"{v1:.0f}"
        if isinstance(v2, float):
            v2 = f"{v2:.0f}"
        if file3 and isinstance(v3, float):
            v3 = f"{v3:.0f}"

        if file3:
            print(f"{key:<25} {str(v1):<20} {str(v2):<20} {str(v3):<20}")
        else:
            print(f"{key:<25} {str(v1):<30} {str(v2):<30}")

    # Evaluation time
    time1 = data1.get("total_evaluation_time_seconds", "N/A")
    time2 = data2.get("total_evaluation_time_seconds", "N/A")
    time3 = data3.get("total_evaluation_time_seconds", "N/A") if data3 else None
    if time1 != "N/A":
        time1 = f"{float(time1):.1f}s"
    if time2 != "N/A":
        time2 = f"{float(time2):.1f}s"
    if file3 and time3 != "N/A":
        time3 = f"{float(time3):.1f}s"

    if file3:
        print(f"{'eval_time':<25} {time1:<20} {time2:<20} {time3:<20}")
    else:
        print(f"{'eval_time':<25} {time1:<30} {time2:<30}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two or three lm_eval JSON result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_evals.py base.json abliterated.json
  python compare_evals.py base.json model_a.json model_b.json  # Compare two models against base
  python compare_evals.py --all base.json abliterated.json     # Include stderr
        """
    )
    parser.add_argument("file1", help="Base/reference results JSON file")
    parser.add_argument("file2", help="Second results JSON file (compared against base)")
    parser.add_argument("file3", nargs="?", default=None,
                        help="Optional third results JSON file (compared against base)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all metrics including stderr")

    args = parser.parse_args()

    # Validate files exist
    files_to_check = [args.file1, args.file2]
    if args.file3:
        files_to_check.append(args.file3)
    for f in files_to_check:
        if not Path(f).exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    compare_results(args.file1, args.file2, file3=args.file3, show_all=args.all)


if __name__ == "__main__":
    main()
