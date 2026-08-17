from __future__ import annotations

import argparse
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

from app.adaptation import select_balanced_support_set
from app.cross_domain import incident_to_cross_domain_text
from app.ood import fit_similarity_ood_threshold
from app.risk_control import wilson_upper_bound


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=7,
    help="Predeclared seed for selecting 600 labelled BGL support incidents.",
)
args = parser.parse_args()
SEED = args.seed

BGL_DEVELOPMENT_PATH = ROOT / "data/processed/bgl_development_windows.jsonl"
LOCKED_HOLDOUT_PATH = (
    ROOT / "data/processed/bgl_locked_temporal_holdout.jsonl"
)

ARTIFACT_DIR = ROOT / "artifacts/cross_domain"
RESULTS_PATH = (
    ARTIFACT_DIR / f"bgl_locked_holdout_final_seed_{SEED}.json"
)

SUPPORT_SIZE = 600
TOP_K = 3
MAX_VALIDATION_UNSAFE_RATE = 0.05
ALLOWED_TARGET_OOD_RATE = 0.05
FAMILYWISE_RISK_CONFIDENCE = 0.95
MIN_VALIDATION_AUTOMATED = 100

BGL_WINDOW_NUMBER = re.compile(r"bgl-window-(\d+)$")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def bgl_time_key(incident: dict) -> int:
    match = BGL_WINDOW_NUMBER.search(str(incident["incident_id"]))

    if not match:
        raise ValueError(f"Invalid BGL incident ID: {incident['incident_id']}")

    return int(match.group(1))


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


def select_anomaly_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
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


def retrieve_top_evidence(
    query_matrix,
    support_matrix,
    support_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = (query_matrix @ support_matrix.T).toarray()

    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = np.take_along_axis(
        similarities,
        top_indices,
        axis=1,
    )
    top_labels = support_labels[top_indices]

    return top_scores[:, 0], top_labels


def calibrate_safety_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    ood_cutoff: float,
) -> dict[str, float | int]:
    """
    Select the highest-coverage policy whose Bonferroni-adjusted,
    one-sided Wilson upper confidence bound is at most 5%.
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
    agreement = evidence_labels[:, 0] == predictions

    confidence_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]

    similarity_thresholds = sorted(
        {
            float(ood_cutoff),
            *[
                float(np.quantile(maximum_similarity, quantile))
                for quantile in [0.10, 0.20, 0.30, 0.40, 0.50]
            ],
        }
    )

    candidate_count = (
        len(confidence_thresholds) * len(similarity_thresholds)
    )
    per_policy_confidence = 1.0 - (
        (1.0 - FAMILYWISE_RISK_CONFIDENCE) / candidate_count
    )

    best_policy = None

    for confidence_threshold in confidence_thresholds:
        for similarity_threshold in similarity_thresholds:
            automated = (
                (confidence >= confidence_threshold)
                & (maximum_similarity >= similarity_threshold)
                & unanimous
                & agreement
            )

            automated_count = int(automated.sum())

            if automated_count < MIN_VALIDATION_AUTOMATED:
                continue

            errors = int(
                (predictions[automated] != y_true[automated]).sum()
            )
            observed_unsafe_rate = errors / automated_count

            unsafe_upper_bound = wilson_upper_bound(
                errors=errors,
                total=automated_count,
                confidence=per_policy_confidence,
            )

            if unsafe_upper_bound <= MAX_VALIDATION_UNSAFE_RATE:
                candidate = {
                    "confidence_threshold": confidence_threshold,
                    "similarity_threshold": similarity_threshold,
                    "coverage": float(automated.mean()),
                    "automated_accuracy": 1.0 - observed_unsafe_rate,
                    "unsafe_rate": observed_unsafe_rate,
                    "validation_errors": errors,
                    "validation_automated_incidents": automated_count,
                    "validation_unsafe_upper_bound": unsafe_upper_bound,
                    "per_policy_confidence": per_policy_confidence,
                }

                if (
                    best_policy is None
                    or candidate["coverage"] > best_policy["coverage"]
                ):
                    best_policy = candidate

    if best_policy is None:
        raise RuntimeError(
            "No validation policy met the conservative risk-control target."
        )

    return best_policy


def selective_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    policy: dict[str, float],
) -> dict[str, float | int | None]:
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
    agreement = evidence_labels[:, 0] == predictions

    automated = (
        (confidence >= policy["confidence_threshold"])
        & (maximum_similarity >= policy["similarity_threshold"])
        & unanimous
        & agreement
    )

    automated_count = int(automated.sum())

    if automated_count:
        automated_accuracy = float(
            (predictions[automated] == y_true[automated]).mean()
        )
        unsafe_count = int(
            (predictions[automated] != y_true[automated]).sum()
        )
        unsafe_rate = float(1.0 - automated_accuracy)
    else:
        automated_accuracy = None
        unsafe_count = 0
        unsafe_rate = None

    return {
        "total_incidents": len(y_true),
        "automated_incidents": automated_count,
        "abstained_incidents": len(y_true) - automated_count,
        "coverage": float(automated.mean()),
        "automated_accuracy": automated_accuracy,
        "unsafe_decision_rate": unsafe_rate,
        "unsafe_decisions": unsafe_count,
        "mean_maximum_similarity": float(maximum_similarity.mean()),
    }


def format_optional_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> None:
    development_incidents = load_jsonl(BGL_DEVELOPMENT_PATH)
    locked_holdout = load_jsonl(LOCKED_HOLDOUT_PATH)

    development_incidents.sort(key=bgl_time_key)
    locked_holdout.sort(key=bgl_time_key)

    if max(map(bgl_time_key, development_incidents)) >= min(
        map(bgl_time_key, locked_holdout)
    ):
        raise RuntimeError(
            "Locked holdout is not temporally later than development data."
        )

    support_end = int(len(development_incidents) * 0.60)
    validation_end = int(len(development_incidents) * 0.80)

    support_pool = development_incidents[:support_end]
    validation_incidents = development_incidents[
        support_end:validation_end
    ]

    support_incidents = select_balanced_support_set(
        support_pool,
        total_size=SUPPORT_SIZE,
        seed=SEED,
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )

    x_support = vectorizer.fit_transform(texts(support_incidents))
    x_validation = vectorizer.transform(texts(validation_incidents))
    x_locked = vectorizer.transform(texts(locked_holdout))

    y_support = labels(support_incidents)
    y_validation = labels(validation_incidents)
    y_locked = labels(locked_holdout)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=7,
    )
    classifier.fit(x_support, y_support)

    validation_probabilities = classifier.predict_proba(
        x_validation
    )[:, 1]

    anomaly_threshold, validation_f1 = select_anomaly_threshold(
        y_true=y_validation,
        probabilities=validation_probabilities,
    )

    validation_similarity, validation_evidence_labels = (
        retrieve_top_evidence(
            query_matrix=x_validation,
            support_matrix=x_support,
            support_labels=y_support,
        )
    )

    ood_cutoff = fit_similarity_ood_threshold(
        validation_similarity,
        allowed_source_ood_rate=ALLOWED_TARGET_OOD_RATE,
    )

    safety_policy = calibrate_safety_policy(
        y_true=y_validation,
        probabilities=validation_probabilities,
        maximum_similarity=validation_similarity,
        evidence_labels=validation_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        ood_cutoff=ood_cutoff,
    )

    locked_probabilities = classifier.predict_proba(x_locked)[:, 1]

    locked_similarity, locked_evidence_labels = retrieve_top_evidence(
        query_matrix=x_locked,
        support_matrix=x_support,
        support_labels=y_support,
    )

    no_abstention = classification_metrics(
        y_true=y_locked,
        probabilities=locked_probabilities,
        threshold=anomaly_threshold,
    )

    selective = selective_metrics(
        y_true=y_locked,
        probabilities=locked_probabilities,
        maximum_similarity=locked_similarity,
        evidence_labels=locked_evidence_labels,
        anomaly_threshold=anomaly_threshold,
        policy=safety_policy,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        ARTIFACT_DIR
        / f"bgl_locked_holdout_support_{SUPPORT_SIZE}_seed_{SEED}.joblib"
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "support_size": SUPPORT_SIZE,
            "support_selection_seed": SEED,
            "anomaly_threshold": anomaly_threshold,
            "ood_cutoff": ood_cutoff,
            "safety_policy": safety_policy,
            "training_domain": "BGL",
        },
        model_path,
    )

    results = {
        "experiment": "bgl_locked_holdout_final_evaluation",
        "support_selection_seed": SEED,
        "support_size": SUPPORT_SIZE,
        "development_support_pool_incidents": len(support_pool),
        "development_validation_incidents": len(validation_incidents),
        "locked_holdout_incidents": len(locked_holdout),
        "validation_f1": validation_f1,
        "anomaly_threshold": anomaly_threshold,
        "ood_cutoff": ood_cutoff,
        "safety_policy": safety_policy,
        "locked_holdout_no_abstention": no_abstention,
        "locked_holdout_selective": selective,
        "model_path": str(model_path),
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("BGL Locked-Holdout Final Evaluation")
    print("=" * 40)
    print(f"Support-selection seed: {SEED}")
    print(f"Support incidents: {SUPPORT_SIZE}")
    print(f"Locked holdout incidents: {len(locked_holdout):,}")
    print(f"Validation F1: {validation_f1:.4f}")
    print(f"Anomaly threshold: {anomaly_threshold:.2f}")
    print(
        "Safety policy: "
        f"confidence>={safety_policy['confidence_threshold']:.2f}, "
        f"similarity>={safety_policy['similarity_threshold']:.4f}"
    )
    print(
        "Validation unsafe-rate upper bound: "
        f"{safety_policy['validation_unsafe_upper_bound']:.4f}"
    )
    print()
    print("Locked holdout without abstention:")
    for key, value in no_abstention.items():
        print(f"  {key}: {value:.4f}")

    print("Locked holdout with abstention:")
    for key, value in selective.items():
        if value is None:
            print(f"  {key}: N/A")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSaved final results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()