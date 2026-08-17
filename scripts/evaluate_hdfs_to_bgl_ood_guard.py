import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cross_domain import incident_to_cross_domain_text
from app.ood import (
    fit_similarity_ood_threshold,
    is_in_distribution,
)


HDFS_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
BGL_PATH = ROOT / "data/processed/bgl_windows.jsonl"

MODEL_PATH = (
    ROOT / "artifacts/cross_domain"
    / "hdfs_source_logistic_regression.joblib"
)
RESULTS_PATH = (
    ROOT / "artifacts/cross_domain"
    / "hdfs_to_bgl_ood_guard_results.json"
)

TOP_K = 3
ALLOWED_SOURCE_OOD_RATE = 0.05

HDFS_TIME = re.compile(r"^(\d{6})\s+(\d{6})\b")
BGL_WINDOW_NUMBER = re.compile(r"bgl-window-(\d+)$")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hdfs_time_key(incident: dict) -> str:
    for event in incident.get("events", []):
        match = HDFS_TIME.match(str(event.get("message", "")))

        if match:
            return match.group(1) + match.group(2)

    raise ValueError(f"No HDFS timestamp for {incident['incident_id']}")


def bgl_time_key(incident: dict) -> int:
    match = BGL_WINDOW_NUMBER.search(str(incident["incident_id"]))

    if not match:
        raise ValueError(f"Invalid BGL incident ID: {incident['incident_id']}")

    return int(match.group(1))


def chronological_split(
    incidents: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    train_end = int(len(incidents) * 0.60)
    validation_end = int(len(incidents) * 0.80)

    return (
        incidents[:train_end],
        incidents[train_end:validation_end],
        incidents[validation_end:],
    )


def labels(incidents: list[dict]) -> np.ndarray:
    return np.array(
        [int(incident["is_anomaly"]) for incident in incidents]
    )


def texts(incidents: list[dict]) -> list[str]:
    return [
        incident_to_cross_domain_text(incident)
        for incident in incidents
    ]


def retrieve_top_evidence(
    query_matrix,
    train_matrix,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum source similarity and top-3 source evidence labels."""
    similarities = (query_matrix @ train_matrix.T).toarray()

    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = np.take_along_axis(
        similarities,
        top_indices,
        axis=1,
    )
    top_labels = train_labels[top_indices]

    return top_scores[:, 0], top_labels


def base_abstention_mask(
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    policy: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the original HDFS-calibrated evidence agreement policy.
    """
    predictions = (probabilities >= anomaly_threshold).astype(int)

    confidence = np.where(
        predictions == 1,
        probabilities,
        1.0 - probabilities,
    )

    unanimous = np.all(
        evidence_labels == evidence_labels[:, [0]],
        axis=1,
    )
    evidence_prediction = evidence_labels[:, 0]
    agreement = evidence_prediction == predictions

    automated = (
        (confidence >= float(policy["confidence_threshold"]))
        & (
            maximum_similarity
            >= float(policy["similarity_threshold"])
        )
        & unanimous
        & agreement
    )

    return automated, predictions


def evaluate_mask(
    y_true: np.ndarray,
    predictions: np.ndarray,
    automated: np.ndarray,
    maximum_similarity: np.ndarray,
) -> dict[str, float | int]:
    automated_count = int(automated.sum())
    total_count = len(y_true)

    if automated_count:
        automated_accuracy = float(
            (predictions[automated] == y_true[automated]).mean()
        )
        unsafe_count = int(
            (predictions[automated] != y_true[automated]).sum()
        )
    else:
        automated_accuracy = 0.0
        unsafe_count = 0

    return {
        "total_incidents": total_count,
        "automated_incidents": automated_count,
        "abstained_incidents": total_count - automated_count,
        "coverage": float(automated.mean()),
        "automated_accuracy": automated_accuracy,
        "unsafe_decision_rate": float(1.0 - automated_accuracy)
        if automated_count
        else 0.0,
        "unsafe_decisions": unsafe_count,
        "mean_maximum_hdfs_similarity": float(
            maximum_similarity.mean()
        ),
        "median_maximum_hdfs_similarity": float(
            np.median(maximum_similarity)
        ),
    }


def print_metrics(title: str, metrics: dict[str, float | int]) -> None:
    print()
    print(title)
    print("-" * len(title))

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Missing source model. Run "
            "scripts/evaluate_hdfs_to_bgl_transfer.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    anomaly_threshold = float(artifact["anomaly_threshold"])
    base_policy = artifact["abstention_policy"]

    hdfs_incidents = load_jsonl(HDFS_PATH)
    bgl_incidents = load_jsonl(BGL_PATH)

    hdfs_incidents.sort(key=hdfs_time_key)
    bgl_incidents.sort(key=bgl_time_key)

    hdfs_train, hdfs_validation, hdfs_test = chronological_split(
        hdfs_incidents
    )
    _, _, bgl_target_test = chronological_split(bgl_incidents)

    x_hdfs_train = vectorizer.transform(texts(hdfs_train))
    x_hdfs_validation = vectorizer.transform(texts(hdfs_validation))
    x_hdfs_test = vectorizer.transform(texts(hdfs_test))
    x_bgl_target = vectorizer.transform(texts(bgl_target_test))

    y_hdfs_train = labels(hdfs_train)
    y_hdfs_test = labels(hdfs_test)
    y_bgl_target = labels(bgl_target_test)

    # Fit the OOD cutoff using HDFS validation similarity only.
    validation_similarity, _ = retrieve_top_evidence(
        query_matrix=x_hdfs_validation,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )

    source_similarity_cutoff = fit_similarity_ood_threshold(
        validation_similarity,
        allowed_source_ood_rate=ALLOWED_SOURCE_OOD_RATE,
    )

    # In-domain HDFS reference.
    hdfs_test_probabilities = classifier.predict_proba(x_hdfs_test)[:, 1]
    hdfs_test_similarity, hdfs_test_evidence_labels = retrieve_top_evidence(
        query_matrix=x_hdfs_test,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )
    hdfs_base_automated, hdfs_predictions = base_abstention_mask(
        probabilities=hdfs_test_probabilities,
        maximum_similarity=hdfs_test_similarity,
        evidence_labels=hdfs_test_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        policy=base_policy,
    )
    hdfs_ood_automated = hdfs_base_automated & is_in_distribution(
        hdfs_test_similarity,
        similarity_threshold=source_similarity_cutoff,
    )

    # Zero-shot target BGL evaluation.
    bgl_probabilities = classifier.predict_proba(x_bgl_target)[:, 1]
    bgl_similarity, bgl_evidence_labels = retrieve_top_evidence(
        query_matrix=x_bgl_target,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )
    bgl_base_automated, bgl_predictions = base_abstention_mask(
        probabilities=bgl_probabilities,
        maximum_similarity=bgl_similarity,
        evidence_labels=bgl_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        policy=base_policy,
    )
    bgl_ood_automated = bgl_base_automated & is_in_distribution(
        bgl_similarity,
        similarity_threshold=source_similarity_cutoff,
    )

    hdfs_base_results = evaluate_mask(
        y_true=y_hdfs_test,
        predictions=hdfs_predictions,
        automated=hdfs_base_automated,
        maximum_similarity=hdfs_test_similarity,
    )
    hdfs_ood_results = evaluate_mask(
        y_true=y_hdfs_test,
        predictions=hdfs_predictions,
        automated=hdfs_ood_automated,
        maximum_similarity=hdfs_test_similarity,
    )
    bgl_base_results = evaluate_mask(
        y_true=y_bgl_target,
        predictions=bgl_predictions,
        automated=bgl_base_automated,
        maximum_similarity=bgl_similarity,
    )
    bgl_ood_results = evaluate_mask(
        y_true=y_bgl_target,
        predictions=bgl_predictions,
        automated=bgl_ood_automated,
        maximum_similarity=bgl_similarity,
    )

    results = {
        "experiment": "hdfs_to_bgl_ood_aware_abstention",
        "source_domain": "HDFS",
        "target_domain": "BGL",
        "source_ood_calibration": {
            "allowed_source_ood_rate": ALLOWED_SOURCE_OOD_RATE,
            "source_similarity_cutoff": source_similarity_cutoff,
            "calibration_data": "HDFS validation only",
        },
        "hdfs_in_domain": {
            "base_evidence_abstention": hdfs_base_results,
            "ood_aware_abstention": hdfs_ood_results,
        },
        "bgl_zero_shot": {
            "base_evidence_abstention": bgl_base_results,
            "ood_aware_abstention": bgl_ood_results,
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("HDFS-to-BGL OOD-Aware Abstention")
    print("=" * 36)
    print(
        "OOD cutoff selected from HDFS validation only: "
        f"{source_similarity_cutoff:.4f}"
    )
    print(
        "Allowed HDFS validation OOD rate: "
        f"{ALLOWED_SOURCE_OOD_RATE:.2%}"
    )

    print_metrics(
        "HDFS In-Domain: Original Evidence Abstention",
        hdfs_base_results,
    )
    print_metrics(
        "HDFS In-Domain: OOD-Aware Abstention",
        hdfs_ood_results,
    )
    print_metrics(
        "BGL Zero-Shot: Original Evidence Abstention",
        bgl_base_results,
    )
    print_metrics(
        "BGL Zero-Shot: OOD-Aware Abstention",
        bgl_ood_results,
    )

    print()
    print(f"Saved OOD results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()