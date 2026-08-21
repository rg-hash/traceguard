from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from app.investigation import (
    ROOT_CAUSES,
    events_to_text,
    get_investigation_service,
)


BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "investigation_benchmark.json"
)


def keyword_baseline(
    events: list[dict],
) -> str | None:
    """
    Predict a root-cause category using only keyword counts.

    This baseline has no incident retrieval, deployment context,
    runbooks, or safety-aware investigation workflow.
    """
    log_text = events_to_text(events)

    scores = {
        root_cause: sum(
            keyword in log_text
            for keyword in keywords
        )
        for root_cause, keywords in ROOT_CAUSES.items()
    }

    predicted_cause = max(
        scores,
        key=scores.get,
    )

    if scores[predicted_cause] == 0:
        return None

    return predicted_cause


def retrieval_only_baseline(
    events: list[dict],
) -> str | None:
    """
    Predict from the most similar historical INCIDENT only.

    Runbooks are skipped because they describe checks rather than
    confirmed root causes.
    """
    service = get_investigation_service()

    query = events_to_text(events)

    query_vector = service.vectorizer.transform([query])

    scores = (
        query_vector
        @ service.knowledge_matrix.T
    ).toarray().ravel()

    for index in scores.argsort()[::-1]:
        item = service.knowledge[index]

        if item["kind"] == "incident":
            return item.get("root_cause")

    return None


def full_traceguard_prediction(
    events: list[dict],
    incident_id: str,
    triage_recommendation: str,
) -> str | None:
    """Return the highest-ranked full-agent hypothesis."""
    service = get_investigation_service()

    result = service.investigate(
        incident_id=incident_id,
        events=events,
        triage_recommendation=triage_recommendation,
        triage_context={},
    )

    hypotheses = result["hypotheses"]

    if not hypotheses:
        return None

    return hypotheses[0]["cause"]


def accuracy(
    predictions: list[str | None],
    expected: list[str],
) -> float:
    correct = sum(
        prediction == label
        for prediction, label in zip(
            predictions,
            expected,
        )
    )

    return correct / len(expected)


def evaluate() -> dict:
    benchmark = json.loads(
        BENCHMARK_PATH.read_text()
    )

    expected = [
        case["expected_root_cause"]
        for case in benchmark
    ]

    keyword_predictions = []
    retrieval_predictions = []
    traceguard_predictions = []

    per_case = []

    for case in benchmark:
        keyword_prediction = keyword_baseline(
            case["events"]
        )

        retrieval_prediction = retrieval_only_baseline(
            case["events"]
        )

        traceguard_prediction = full_traceguard_prediction(
            events=case["events"],
            incident_id=case["incident_id"],
            triage_recommendation=(
                case["triage_recommendation"]
            ),
        )

        keyword_predictions.append(keyword_prediction)

        retrieval_predictions.append(retrieval_prediction)

        traceguard_predictions.append(traceguard_prediction)

        per_case.append(
            {
                "incident_id": case["incident_id"],
                "expected": case[
                    "expected_root_cause"
                ],
                "keyword": keyword_prediction,
                "retrieval_only": retrieval_prediction,
                "traceguard": traceguard_prediction,
            }
        )

    return {
        "total_cases": len(benchmark),
        "keyword_accuracy": accuracy(
            keyword_predictions,
            expected,
        ),
        "retrieval_only_accuracy": accuracy(
            retrieval_predictions,
            expected,
        ),
        "traceguard_accuracy": accuracy(
            traceguard_predictions,
            expected,
        ),
        "per_case": per_case,
    }


if __name__ == "__main__":
    results = evaluate()

    print(
        f"Cases: {results['total_cases']}"
    )

    print(
        "Keyword-only Top-1 accuracy: "
        f"{results['keyword_accuracy']:.2%}"
    )

    print(
        "Retrieval-only Top-1 accuracy: "
        f"{results['retrieval_only_accuracy']:.2%}"
    )

    print(
        "Full TraceGuard Top-1 accuracy: "
        f"{results['traceguard_accuracy']:.2%}"
    )

    print("\nPer-case predictions:")

    for item in results["per_case"]:
        print(
            f"- {item['incident_id']} | "
            f"expected={item['expected']} | "
            f"keyword={item['keyword']} | "
            f"retrieval={item['retrieval_only']} | "
            f"traceguard={item['traceguard']}"
        )