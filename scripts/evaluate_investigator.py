from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from app.investigation import get_investigation_service


BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "investigation_benchmark.json"
)


def evaluate() -> dict:
    benchmark = json.loads(
        BENCHMARK_PATH.read_text()
    )

    service = get_investigation_service()

    top_1_correct = 0
    top_3_correct = 0
    evidence_hits = 0
    safe_decisions = 0

    results = []

    for case in benchmark:
        result = service.investigate(
            incident_id=case["incident_id"],
            events=case["events"],
            triage_recommendation=(
                case["triage_recommendation"]
            ),
            triage_context={},
        )

        predicted_causes = [
            hypothesis["cause"]
            for hypothesis in result["hypotheses"]
        ]

        evidence_ids = [
            evidence["id"]
            for evidence in result["evidence"]
        ]

        top_1_hit = (
            bool(predicted_causes)
            and predicted_causes[0]
            == case["expected_root_cause"]
        )

        top_3_hit = (
            case["expected_root_cause"]
            in predicted_causes
        )

        evidence_hit = (
            case["expected_evidence_id"]
            in evidence_ids
        )

        safe_decision = (
            result["decision"]
            == "ENGINEER_REVIEW_REQUIRED"
        )

        top_1_correct += top_1_hit
        top_3_correct += top_3_hit
        evidence_hits += evidence_hit
        safe_decisions += safe_decision

        results.append(
            {
                "incident_id": case["incident_id"],
                "expected_root_cause": (
                    case["expected_root_cause"]
                ),
                "predicted_causes": predicted_causes,
                "evidence_ids": evidence_ids,
                "top_1_correct": top_1_hit,
                "top_3_correct": top_3_hit,
                "evidence_hit": evidence_hit,
                "safe_decision": safe_decision,
            }
        )

    total = len(benchmark)

    return {
        "total_cases": total,
        "top_1_root_cause_accuracy": (
            top_1_correct / total
        ),
        "top_3_root_cause_accuracy": (
            top_3_correct / total
        ),
        "evidence_recall_at_4": evidence_hits / total,
        "safe_decision_rate": safe_decisions / total,
        "results": results,
    }


if __name__ == "__main__":
    report = evaluate()

    print(
        "Top-1 root-cause accuracy: "
        f"{report['top_1_root_cause_accuracy']:.2%}"
    )

    print(
        "Top-3 root-cause accuracy: "
        f"{report['top_3_root_cause_accuracy']:.2%}"
    )

    print(
        "Evidence Recall@4: "
        f"{report['evidence_recall_at_4']:.2%}"
    )

    print(
        "Safe decision rate: "
        f"{report['safe_decision_rate']:.2%}"
    )

    print("\nPer-case results:")

    for item in report["results"]:
        status = (
            "PASS"
            if item["top_1_correct"]
            and item["evidence_hit"]
            else "CHECK"
        )

        print(
            f"- {status}: {item['incident_id']} | "
            f"predicted={item['predicted_causes']} | "
            f"evidence={item['evidence_ids']}"
        )