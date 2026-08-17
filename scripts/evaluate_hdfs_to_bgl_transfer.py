import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cross_domain import incident_to_cross_domain_text


HDFS_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
BGL_PATH = ROOT / "data/processed/bgl_windows.jsonl"

ARTIFACT_DIR = ROOT / "artifacts/cross_domain"
MODEL_PATH = ARTIFACT_DIR / "hdfs_source_logistic_regression.joblib"
RESULTS_PATH = ARTIFACT_DIR / "hdfs_to_bgl_zero_shot_results.json"

TOP_K = 3
MAX_VALIDATION_UNSAFE_RATE = 0.05

HDFS_TIME = re.compile(r"^(\d{6})\s+(\d{6})\b")
BGL_WINDOW_NUMBER = re.compile(r"bgl-window-(\d+)$")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hdfs_time_key(incident: dict) -> str:
    """Extract chronological order from the first HDFS event."""
    for event in incident.get("events", []):
        match = HDFS_TIME.match(str(event.get("message", "")))
        if match:
            return match.group(1) + match.group(2)

    raise ValueError(f"No HDFS timestamp for {incident['incident_id']}")


def bgl_time_key(incident: dict) -> int:
    """BGL window numbers preserve order in the original log stream."""
    match = BGL_WINDOW_NUMBER.search(str(incident["incident_id"]))

    if not match:
        raise ValueError(f"Invalid BGL incident ID: {incident['incident_id']}")

    return int(match.group(1))


def chronological_split(
    incidents: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Create 60% train, 20% validation, 20% final-test partitions."""
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


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def top_evidence(
    query_matrix,
    train_matrix,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve top-k source-domain evidence using source TF-IDF vectors.

    Returns:
    - maximum lexical similarity per query;
    - top-k historic evidence labels;
    - top-k evidence indexes.
    """
    similarities = (query_matrix @ train_matrix.T).toarray()

    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = np.take_along_axis(
        similarities,
        top_indices,
        axis=1,
    )
    top_labels = train_labels[top_indices]

    return top_scores[:, 0], top_labels, top_indices


def calibrate_abstention_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
) -> dict[str, float | bool]:
    """
    Select policy on HDFS validation data only.

    Automation requires:
    - classifier confidence;
    - similarity to historic HDFS evidence;
    - unanimous top-3 evidence;
    - evidence label agrees with classifier prediction.
    """
    predictions = (probabilities >= anomaly_threshold).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)

    unanimous = np.all(
        evidence_labels == evidence_labels[:, [0]],
        axis=1,
    )
    evidence_prediction = evidence_labels[:, 0]
    agreement = evidence_prediction == predictions

    best_policy = None

    confidence_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]
    similarity_thresholds = [
        0.00, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70,
    ]

    for confidence_threshold in confidence_thresholds:
        for similarity_threshold in similarity_thresholds:
            automated = (
                (confidence >= confidence_threshold)
                & (maximum_similarity >= similarity_threshold)
                & unanimous
                & agreement
            )

            if not automated.any():
                continue

            automated_accuracy = (
                predictions[automated] == y_true[automated]
            ).mean()

            unsafe_rate = 1.0 - automated_accuracy
            coverage = automated.mean()

            if unsafe_rate <= MAX_VALIDATION_UNSAFE_RATE:
                candidate = {
                    "confidence_threshold": confidence_threshold,
                    "similarity_threshold": similarity_threshold,
                    "coverage": float(coverage),
                    "automated_accuracy": float(automated_accuracy),
                    "unsafe_rate": float(unsafe_rate),
                }

                if (
                    best_policy is None
                    or candidate["coverage"] > best_policy["coverage"]
                ):
                    best_policy = candidate

    if best_policy is None:
        raise RuntimeError(
            "No HDFS validation abstention policy met the safety target."
        )

    return best_policy


def selective_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    policy: dict[str, float | bool],
) -> dict[str, float | int]:
    """Evaluate the fixed abstention policy on any partition or domain."""
    predictions = (probabilities >= anomaly_threshold).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)

    unanimous = np.all(
        evidence_labels == evidence_labels[:, [0]],
        axis=1,
    )
    evidence_prediction = evidence_labels[:, 0]
    agreement = evidence_prediction == predictions

    automated = (
        (confidence >= float(policy["confidence_threshold"]))
        & (maximum_similarity >= float(policy["similarity_threshold"]))
        & unanimous
        & agreement
    )

    automated_count = int(automated.sum())
    total_count = len(y_true)

    if automated_count:
        automated_accuracy = float(
            (predictions[automated] == y_true[automated]).mean()
        )
        unsafe_rate = float(1.0 - automated_accuracy)
    else:
        automated_accuracy = 0.0
        unsafe_rate = 0.0

    return {
        "total_incidents": total_count,
        "automated_incidents": automated_count,
        "abstained_incidents": total_count - automated_count,
        "coverage": float(automated.mean()),
        "automated_accuracy": automated_accuracy,
        "unsafe_decision_rate": unsafe_rate,
        "unsafe_decisions": int(
            (predictions[automated] != y_true[automated]).sum()
        ),
        "mean_maximum_hdfs_similarity": float(maximum_similarity.mean()),
        "zero_feature_rate": float(
            (maximum_similarity == 0.0).mean()
        ),
    }


def select_anomaly_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Select the HDFS validation F1-optimal anomaly threshold."""
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.05):
        metrics = classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        if metrics["f1"] > best_f1:
            best_threshold = float(threshold)
            best_f1 = metrics["f1"]

    return best_threshold, best_f1


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
    # 1. Load and sort each domain independently.
    hdfs_incidents = load_jsonl(HDFS_PATH)
    bgl_incidents = load_jsonl(BGL_PATH)

    hdfs_incidents.sort(key=hdfs_time_key)
    bgl_incidents.sort(key=bgl_time_key)

    hdfs_train, hdfs_validation, hdfs_test = chronological_split(
        hdfs_incidents
    )

    # BGL is never used for fitting or policy selection.
    _, _, bgl_target_test = chronological_split(bgl_incidents)

    # 2. Fit vectorizer and classifier using HDFS training data only.
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )

    hdfs_train_texts = texts(hdfs_train)
    hdfs_validation_texts = texts(hdfs_validation)
    hdfs_test_texts = texts(hdfs_test)
    bgl_target_texts = texts(bgl_target_test)

    x_hdfs_train = vectorizer.fit_transform(hdfs_train_texts)
    x_hdfs_validation = vectorizer.transform(hdfs_validation_texts)
    x_hdfs_test = vectorizer.transform(hdfs_test_texts)
    x_bgl_target = vectorizer.transform(bgl_target_texts)

    y_hdfs_train = labels(hdfs_train)
    y_hdfs_validation = labels(hdfs_validation)
    y_hdfs_test = labels(hdfs_test)
    y_bgl_target = labels(bgl_target_test)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=7,
    )
    classifier.fit(x_hdfs_train, y_hdfs_train)

    # 3. Use HDFS validation only to select classification threshold.
    validation_probabilities = classifier.predict_proba(
        x_hdfs_validation
    )[:, 1]

    anomaly_threshold, validation_f1 = select_anomaly_threshold(
        y_true=y_hdfs_validation,
        probabilities=validation_probabilities,
    )

    # 4. Use HDFS validation only to select abstention policy.
    validation_similarity, validation_evidence_labels, _ = top_evidence(
        query_matrix=x_hdfs_validation,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )

    abstention_policy = calibrate_abstention_policy(
        y_true=y_hdfs_validation,
        probabilities=validation_probabilities,
        maximum_similarity=validation_similarity,
        evidence_labels=validation_evidence_labels,
        anomaly_threshold=anomaly_threshold,
    )

    # 5. Evaluate frozen source model on future HDFS reference data.
    hdfs_test_probabilities = classifier.predict_proba(x_hdfs_test)[:, 1]
    hdfs_test_similarity, hdfs_test_evidence_labels, _ = top_evidence(
        query_matrix=x_hdfs_test,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )

    hdfs_reference_classification = classification_metrics(
        y_true=y_hdfs_test,
        probabilities=hdfs_test_probabilities,
        threshold=anomaly_threshold,
    )
    hdfs_reference_selective = selective_metrics(
        y_true=y_hdfs_test,
        probabilities=hdfs_test_probabilities,
        maximum_similarity=hdfs_test_similarity,
        evidence_labels=hdfs_test_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        policy=abstention_policy,
    )

    # 6. Zero-shot transfer: apply frozen HDFS model to BGL target data.
    bgl_probabilities = classifier.predict_proba(x_bgl_target)[:, 1]
    bgl_similarity, bgl_evidence_labels, _ = top_evidence(
        query_matrix=x_bgl_target,
        train_matrix=x_hdfs_train,
        train_labels=y_hdfs_train,
    )

    bgl_zero_shot_classification = classification_metrics(
        y_true=y_bgl_target,
        probabilities=bgl_probabilities,
        threshold=anomaly_threshold,
    )
    bgl_zero_shot_selective = selective_metrics(
        y_true=y_bgl_target,
        probabilities=bgl_probabilities,
        maximum_similarity=bgl_similarity,
        evidence_labels=bgl_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        policy=abstention_policy,
    )

    # 7. Save source model and complete, reproducible result record.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "source_domain": "HDFS",
            "target_domain": "BGL",
            "anomaly_threshold": anomaly_threshold,
            "abstention_policy": abstention_policy,
            "normalization": "cross_domain_v1",
        },
        MODEL_PATH,
    )

    results = {
        "experiment": "hdfs_to_bgl_zero_shot_transfer",
        "source_domain": "HDFS",
        "target_domain": "BGL",
        "source_split": "chronological_60_20_20",
        "target_partition": "final_20_percent_chronological",
        "hdfs_counts": {
            "train": len(hdfs_train),
            "validation": len(hdfs_validation),
            "final_test": len(hdfs_test),
        },
        "bgl_target_count": len(bgl_target_test),
        "selected_anomaly_threshold": anomaly_threshold,
        "validation_f1_at_selected_threshold": validation_f1,
        "selected_abstention_policy": abstention_policy,
        "hdfs_in_domain_reference": {
            "classification": hdfs_reference_classification,
            "selective": hdfs_reference_selective,
        },
        "bgl_zero_shot_transfer": {
            "classification": bgl_zero_shot_classification,
            "selective": bgl_zero_shot_selective,
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("HDFS-to-BGL Zero-Shot Transfer Experiment")
    print("=" * 44)
    print(f"HDFS train incidents: {len(hdfs_train):,}")
    print(f"HDFS validation incidents: {len(hdfs_validation):,}")
    print(f"HDFS final test incidents: {len(hdfs_test):,}")
    print(f"BGL target test incidents: {len(bgl_target_test):,}")
    print()
    print(
        f"Selected HDFS anomaly threshold: {anomaly_threshold:.2f} "
        f"(validation F1={validation_f1:.4f})"
    )
    print(
        "Selected abstention policy: "
        f"confidence>={abstention_policy['confidence_threshold']:.2f}, "
        f"HDFS evidence similarity>="
        f"{abstention_policy['similarity_threshold']:.2f}"
    )

    print_metrics(
        "HDFS In-Domain Reference: No Abstention",
        hdfs_reference_classification,
    )
    print_metrics(
        "HDFS In-Domain Reference: With Abstention",
        hdfs_reference_selective,
    )
    print_metrics(
        "BGL Zero-Shot Transfer: No Abstention",
        bgl_zero_shot_classification,
    )
    print_metrics(
        "BGL Zero-Shot Transfer: With HDFS-Calibrated Abstention",
        bgl_zero_shot_selective,
    )

    print()
    print(f"Saved source model: {MODEL_PATH}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()