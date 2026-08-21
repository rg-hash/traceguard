"""Evaluate a fixed Investigator configuration on a labelled dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.investigation import get_investigation_service

DEFAULT_DATASET = PROJECT_ROOT / "data/evaluation/investigation_holdout.json"


def evaluate(dataset_path: Path) -> dict:
    dataset_path = dataset_path.resolve()
    cases = json.loads(dataset_path.read_text())
    service = get_investigation_service()
    known = [case for case in cases if case["expected_root_cause"]]
    unknown = [case for case in cases if not case["expected_root_cause"]]
    top_1 = top_3 = true_hypotheses = emitted_hypotheses = 0
    unknown_abstentions = 0
    rows = []

    for case in cases:
        result = service.investigate(
            incident_id=case["incident_id"],
            events=case["events"],
            triage_recommendation="NEEDS_HUMAN_REVIEW",
            triage_context={},
        )
        predicted = [item["cause"] for item in result["hypotheses"]]
        expected = case["expected_root_cause"]

        if expected:
            top_1 += bool(predicted) and predicted[0] == expected
            top_3 += expected in predicted
            emitted_hypotheses += len(predicted)
            true_hypotheses += sum(cause == expected for cause in predicted)
            status = "correct" if predicted and predicted[0] == expected else "incorrect"
        else:
            abstained = not predicted
            unknown_abstentions += abstained
            status = "safe_abstention" if abstained else "unsupported_hypothesis"

        rows.append({
            "incident_id": case["incident_id"],
            "expected_root_cause": expected,
            "predicted_causes": predicted,
            "status": status,
        })

    return {
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "total_cases": len(cases),
        "known_cases": len(known),
        "unknown_cases": len(unknown),
        "known_top_1_accuracy": top_1 / len(known) if known else 0.0,
        "known_top_3_accuracy": top_3 / len(known) if known else 0.0,
        "known_hypothesis_precision": (
            true_hypotheses / emitted_hypotheses
            if emitted_hypotheses else 0.0
        ),
        "mean_hypotheses_per_known_case": (
            emitted_hypotheses / len(known) if known else 0.0
        ),
        "unknown_abstention_rate": (
            unknown_abstentions / len(unknown) if unknown else 0.0
        ),
        "unknown_unsupported_hypothesis_rate": (
            (len(unknown) - unknown_abstentions) / len(unknown)
            if unknown else 0.0
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    report = evaluate(args.dataset)
    print(f"Dataset: {report['dataset']}")
    print(f"Total cases: {report['total_cases']}")
    print(f"Known-cause Top-1 accuracy: {report['known_top_1_accuracy']:.2%}")
    print(f"Known-cause Top-3 accuracy: {report['known_top_3_accuracy']:.2%}")
    print(f"Known hypothesis precision: {report['known_hypothesis_precision']:.2%}")
    print("Mean hypotheses per known case: " f"{report['mean_hypotheses_per_known_case']:.2f}")
    print(f"Unknown-cause abstention rate: {report['unknown_abstention_rate']:.2%}")
    print("Unknown unsupported-hypothesis rate: " f"{report['unknown_unsupported_hypothesis_rate']:.2%}")
    print("\nPer-case results:")
    for row in report["rows"]:
        print(
            f"- {row['incident_id']} | expected={row['expected_root_cause']} | "
            f"predicted={row['predicted_causes']} | status={row['status']}"
        )
