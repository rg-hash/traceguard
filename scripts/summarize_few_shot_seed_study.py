from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("artifacts/cross_domain")
SEEDS = [7, 21, 42, 84, 123]
SUPPORT_SIZES = [60, 300, 600]

EXPERIMENTS = {
    "HDFS + BGL adaptation": "hdfs_to_bgl_few_shot_adaptation_seed_{seed}.json",
    "BGL-only baseline": "bgl_only_few_shot_baseline_seed_{seed}.json",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return arithmetic mean and sample standard deviation."""
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def load_results(experiment_name: str, filename_pattern: str) -> dict[int, dict]:
    """Load and validate one JSON result file for every required seed."""
    loaded = {}

    for seed in SEEDS:
        path = RESULTS_DIR / filename_pattern.format(seed=seed)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result for {experiment_name}, seed={seed}: {path}"
            )

        with path.open() as file:
            loaded[seed] = json.load(file)

    return loaded


def abstention_metrics_key(run: dict) -> str:
    """
    Adaptation and BGL-only files use different abstention keys.
    Find the one that is not final_test_no_abstention.
    """
    candidates = [
        key
        for key in run
        if key.startswith("final_test_")
        and key != "final_test_no_abstention"
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one final-test abstention metric block; found {candidates}"
        )

    return candidates[0]


def collect_metrics(results_by_seed: dict[int, dict], support_size: int) -> dict[str, list[float]]:
    metrics = {
        "f1": [],
        "pr_auc": [],
        "coverage": [],
        "automated_accuracy": [],
        "unsafe_decision_rate": [],
    }

    for seed, result in results_by_seed.items():
        run = result["runs"][str(support_size)]
        no_abstention = run["final_test_no_abstention"]
        abstention = run[abstention_metrics_key(run)]

        metrics["f1"].append(no_abstention["f1"])
        metrics["pr_auc"].append(no_abstention["pr_auc"])
        metrics["coverage"].append(abstention["coverage"])
        metrics["automated_accuracy"].append(abstention["automated_accuracy"])
        metrics["unsafe_decision_rate"].append(abstention["unsafe_decision_rate"])

    return metrics


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def format_decimal(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    all_results = {
        name: load_results(name, pattern)
        for name, pattern in EXPERIMENTS.items()
    }

    csv_rows = []
    summary = {}

    print("\nFive-Seed HDFS-to-BGL Few-Shot Study")
    print("Seeds:", SEEDS)
    print("Metric format: mean ± sample standard deviation\n")

    for support_size in SUPPORT_SIZES:
        print(f"Support size: {support_size} labelled BGL incidents")
        print("-" * 92)
        print(
            f"{'Method':<24} {'F1':<16} {'PR-AUC':<16} "
            f"{'Coverage':<18} {'Auto accuracy':<18} {'Unsafe rate':<18}"
        )

        summary[str(support_size)] = {}

        for experiment_name, results_by_seed in all_results.items():
            metrics = collect_metrics(results_by_seed, support_size)

            stats = {
                metric: {
                    "mean": mean_std(values)[0],
                    "std": mean_std(values)[1],
                    "per_seed": {
                        str(seed): value
                        for seed, value in zip(SEEDS, values)
                    },
                }
                for metric, values in metrics.items()
            }

            summary[str(support_size)][experiment_name] = stats

            print(
                f"{experiment_name:<24} "
                f"{format_decimal(stats['f1']['mean'], stats['f1']['std']):<16} "
                f"{format_decimal(stats['pr_auc']['mean'], stats['pr_auc']['std']):<16} "
                f"{format_percent(stats['coverage']['mean'], stats['coverage']['std']):<18} "
                f"{format_percent(stats['automated_accuracy']['mean'], stats['automated_accuracy']['std']):<18} "
                f"{format_percent(stats['unsafe_decision_rate']['mean'], stats['unsafe_decision_rate']['std']):<18}"
            )

            for metric_name, metric_stats in stats.items():
                csv_rows.append(
                    {
                        "support_size": support_size,
                        "method": experiment_name,
                        "metric": metric_name,
                        "mean": metric_stats["mean"],
                        "sample_std": metric_stats["std"],
                        **{
                            f"seed_{seed}": metric_stats["per_seed"][str(seed)]
                            for seed in SEEDS
                        },
                    }
                )

        print()

    csv_path = RESULTS_DIR / "few_shot_seed_summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = RESULTS_DIR / "few_shot_seed_summary.json"
    with json_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved CSV summary: {csv_path}")
    print(f"Saved JSON summary: {json_path}")


if __name__ == "__main__":
    main()