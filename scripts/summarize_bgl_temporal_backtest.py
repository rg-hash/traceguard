import json
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path("artifacts/cross_domain")
SEEDS = [7, 21, 42, 84, 123]
FOLDS = ["fold_1", "fold_2", "fold_3"]


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f} ± 0.0000"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def main() -> None:
    results = {}

    for seed in SEEDS:
        path = RESULTS_DIR / f"bgl_temporal_backtest_600_seed_{seed}.json"

        with path.open() as file:
            results[seed] = json.load(file)

    print("\nBGL 600-Label Expanding-Window Backtest Summary")
    print("=" * 54)
    print("Values are mean ± sample standard deviation across five seeds.\n")

    for fold_name in FOLDS:
        f1_scores = []
        pr_aucs = []
        coverages = []
        unsafe_rates = []

        for seed in SEEDS:
            fold = results[seed]["folds"][fold_name]
            no_abstention = fold["evaluation_no_abstention"]
            selective = fold["evaluation_selective"]

            f1_scores.append(no_abstention["f1"])
            pr_aucs.append(no_abstention["pr_auc"])
            coverages.append(selective["coverage"])

            if selective["unsafe_decision_rate"] is not None:
                unsafe_rates.append(selective["unsafe_decision_rate"])

        violations = sum(rate > 0.05 for rate in unsafe_rates)

        print(f"{fold_name}:")
        print(f"  F1:          {mean_std(f1_scores)}")
        print(f"  PR-AUC:      {mean_std(pr_aucs)}")
        print(f"  Coverage:    {mean_std(coverages)}")
        print(f"  Unsafe rate: {mean_std(unsafe_rates)}")
        print(f"  >5% unsafe-rate violations: {violations}/{len(unsafe_rates)}")
        print()


if __name__ == "__main__":
    main()