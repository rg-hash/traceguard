from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path("artifacts/cross_domain")
SEEDS = [7, 21, 42, 84, 123]
SUPPORT_SIZES = [60, 300, 600]


def selective_block(run: dict) -> dict:
    key = next(
        key
        for key in run
        if key.startswith("final_test_")
        and key != "final_test_no_abstention"
    )
    return run[key]


def load_run(filename: str, seed: int, support_size: int) -> dict:
    with (RESULTS_DIR / filename.format(seed=seed)).open() as file:
        result = json.load(file)
    return result["runs"][str(support_size)]


def describe(values: list[float]) -> str:
    return f"{mean(values) * 100:+.2f} pp ± {stdev(values) * 100:.2f} pp"


def main() -> None:
    adaptation_file = "hdfs_to_bgl_few_shot_adaptation_seed_{seed}.json"
    baseline_file = "bgl_only_few_shot_baseline_seed_{seed}.json"

    print("\nPaired Effects: HDFS+BGL Adaptation minus BGL-only Baseline")
    print("Positive coverage/accuracy/F1 is good; negative unsafe rate is good.\n")

    for support_size in SUPPORT_SIZES:
        differences = {
            "f1": [],
            "coverage": [],
            "automated_accuracy": [],
            "unsafe_decision_rate": [],
        }

        print(f"Support size: {support_size}")
        print("Seed   ΔF1       ΔCoverage    ΔAuto accuracy   ΔUnsafe rate")

        for seed in SEEDS:
            adapted = load_run(adaptation_file, seed, support_size)
            baseline = load_run(baseline_file, seed, support_size)

            adapted_selective = selective_block(adapted)
            baseline_selective = selective_block(baseline)

            differences["f1"].append(
                adapted["final_test_no_abstention"]["f1"]
                - baseline["final_test_no_abstention"]["f1"]
            )
            differences["coverage"].append(
                adapted_selective["coverage"]
                - baseline_selective["coverage"]
            )
            differences["automated_accuracy"].append(
                adapted_selective["automated_accuracy"]
                - baseline_selective["automated_accuracy"]
            )
            differences["unsafe_decision_rate"].append(
                adapted_selective["unsafe_decision_rate"]
                - baseline_selective["unsafe_decision_rate"]
            )

            print(
                f"{seed:<6} "
                f"{differences['f1'][-1] * 100:+6.2f} pp   "
                f"{differences['coverage'][-1] * 100:+7.2f} pp   "
                f"{differences['automated_accuracy'][-1] * 100:+7.2f} pp   "
                f"{differences['unsafe_decision_rate'][-1] * 100:+7.2f} pp"
            )

        print("\nMean paired differences:")
        for metric, values in differences.items():
            print(f"  {metric}: {describe(values)}")
        print()


if __name__ == "__main__":
    main()