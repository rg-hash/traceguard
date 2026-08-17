import json
import re
import sys
from pathlib import Path
import argparse
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
from app.ood import (
    fit_similarity_ood_threshold,
    is_in_distribution,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=7,
    help="Random seed for selecting the BGL labelled support set.",
)
args = parser.parse_args()
SEED = args.seed

HDFS_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
BGL_PATH = ROOT / "data/processed/bgl_windows.jsonl"

ARTIFACT_DIR = ROOT / "artifacts/cross_domain"
RESULTS_PATH = (
    ARTIFACT_DIR / f"hdfs_to_bgl_few_shot_adaptation_seed_{SEED}.json"
)

SUPPORT_SIZES = [60, 300, 600]
TOP_K = 3
MAX_VALIDATION_UNSAFE_RATE = 0.05
ALLOWED_TARGET_OOD_RATE = 0.05

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
    train_matrix,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = (query_matrix @ train_matrix.T).toarray()

    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = np.take_along_axis(
        similarities,
        top_indices,
        axis=1,
    )
    top_labels = train_labels[top_indices]

    return top_scores[:, 0], top_labels


def calibrate_safety_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    ood_cutoff: float,
) -> dict[str, float]:
    """
    Select a BGL validation policy after few-shot adaptation.

    The final test partition remains untouched.
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

    best_policy = None

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

            automated_accuracy = float(
                (predictions[automated] == y_true[automated]).mean()
            )
            unsafe_rate = 1.0 - automated_accuracy
            coverage = float(automated.mean())

            if unsafe_rate <= MAX_VALIDATION_UNSAFE_RATE:
                candidate = {
                    "confidence_threshold": confidence_threshold,
                    "similarity_threshold": similarity_threshold,
                    "coverage": coverage,
                    "automated_accuracy": automated_accuracy,
                    "unsafe_rate": unsafe_rate,
                }

                if (
                    best_policy is None
                    or candidate["coverage"] > best_policy["coverage"]
                ):
                    best_policy = candidate

    if best_policy is None:
        raise RuntimeError(
            "No BGL validation safety policy met the unsafe-rate target."
        )

    return best_policy


def selective_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    maximum_similarity: np.ndarray,
    evidence_labels: np.ndarray,
    anomaly_threshold: float,
    policy: dict[str, float],
) -> dict[str, float | int]:
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
    else:
        automated_accuracy = 0.0
        unsafe_count = 0

    return {
        "total_incidents": len(y_true),
        "automated_incidents": automated_count,
        "abstained_incidents": len(y_true) - automated_count,
        "coverage": float(automated.mean()),
        "automated_accuracy": automated_accuracy,
        "unsafe_decision_rate": (
            float(1.0 - automated_accuracy)
            if automated_count
            else 0.0
        ),
        "unsafe_decisions": unsafe_count,
        "mean_maximum_similarity": float(
            maximum_similarity.mean()
        ),
    }


def domain_balanced_weights(
    source_count: int,
    target_count: int,
) -> np.ndarray:
    """
    Give source HDFS and few-shot target BGL data equal total influence.

    Without this, 1,200 HDFS examples would overwhelm 60 BGL support
    examples during adaptation.
    """
    source_weight = 0.5 / source_count
    target_weight = 0.5 / target_count

    weights = np.concatenate(
        [
            np.full(source_count, source_weight),
            np.full(target_count, target_weight),
        ]
    )

    return weights * len(weights)


def print_result(
    support_size: int,
    no_abstention: dict[str, float],
    selective: dict[str, float | int],
    anomaly_threshold: float,
    policy: dict[str, float],
    ood_cutoff: float,
) -> None:
    print(f"Support-selection seed: {SEED}")
    print()
    print(f"BGL Few-Shot Adaptation: {support_size} Labelled Incidents")
    print("-" * 58)
    print(f"BGL threshold selected on validation: {anomaly_threshold:.2f}")
    print(f"BGL OOD cutoff selected on validation: {ood_cutoff:.4f}")
    print(
        "Safety policy: "
        f"confidence>={policy['confidence_threshold']:.2f}, "
        f"similarity>={policy['similarity_threshold']:.4f}"
    )
    print()
    print("No abstention:")
    for key, value in no_abstention.items():
        print(f"  {key}: {value:.4f}")

    print("With adapted abstention:")
    for key, value in selective.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")




def main() -> None:
    hdfs_incidents = load_jsonl(HDFS_PATH)
    bgl_incidents = load_jsonl(BGL_PATH)

    hdfs_incidents.sort(key=hdfs_time_key)
    bgl_incidents.sort(key=bgl_time_key)

    hdfs_train, _, _ = chronological_split(hdfs_incidents)
    bgl_support_pool, bgl_validation, bgl_test = chronological_split(
        bgl_incidents
    )

    hdfs_train_texts = texts(hdfs_train)
    y_hdfs_train = labels(hdfs_train)

    bgl_validation_texts = texts(bgl_validation)
    bgl_test_texts = texts(bgl_test)
    y_bgl_validation = labels(bgl_validation)
    y_bgl_test = labels(bgl_test)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "experiment": "hdfs_to_bgl_few_shot_adaptation",
        "source_domain": "HDFS",
        "target_domain": "BGL",
        "source_training_incidents": len(hdfs_train),
        "bgl_support_pool_incidents": len(bgl_support_pool),
        "bgl_validation_incidents": len(bgl_validation),
        "bgl_final_test_incidents": len(bgl_test),
        "support_sizes": SUPPORT_SIZES,
        "target_split": "chronological_60_20_20",
        "runs": {},
    }

    print("HDFS-to-BGL Few-Shot Adaptation")
    print("=" * 34)
    print(f"HDFS source training incidents: {len(hdfs_train):,}")
    print(f"BGL support pool incidents: {len(bgl_support_pool):,}")
    print(f"BGL validation incidents: {len(bgl_validation):,}")
    print(f"BGL final test incidents: {len(bgl_test):,}")

    for support_size in SUPPORT_SIZES:
        support_incidents = select_balanced_support_set(
            bgl_support_pool,
            total_size=support_size,
            seed=SEED,
        )

        combined_train = hdfs_train + support_incidents
        combined_texts = hdfs_train_texts + texts(support_incidents)
        y_combined_train = np.concatenate(
            [y_hdfs_train, labels(support_incidents)]
        )

        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )

        x_train = vectorizer.fit_transform(combined_texts)
        x_validation = vectorizer.transform(bgl_validation_texts)
        x_test = vectorizer.transform(bgl_test_texts)

        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=7,
        )

        classifier.fit(
            x_train,
            y_combined_train,
            sample_weight=domain_balanced_weights(
                source_count=len(hdfs_train),
                target_count=len(support_incidents),
            ),
        )

        validation_probabilities = classifier.predict_proba(
            x_validation
        )[:, 1]

        anomaly_threshold, validation_f1 = select_anomaly_threshold(
            y_true=y_bgl_validation,
            probabilities=validation_probabilities,
        )

        validation_similarity, validation_evidence_labels = (
            retrieve_top_evidence(
                query_matrix=x_validation,
                train_matrix=x_train,
                train_labels=y_combined_train,
            )
        )

        ood_cutoff = fit_similarity_ood_threshold(
            validation_similarity,
            allowed_source_ood_rate=ALLOWED_TARGET_OOD_RATE,
        )

        safety_policy = calibrate_safety_policy(
            y_true=y_bgl_validation,
            probabilities=validation_probabilities,
            maximum_similarity=validation_similarity,
            evidence_labels=validation_evidence_labels,
            anomaly_threshold=anomaly_threshold,
            ood_cutoff=ood_cutoff,
        )

        test_probabilities = classifier.predict_proba(x_test)[:, 1]
        test_similarity, test_evidence_labels = retrieve_top_evidence(
            query_matrix=x_test,
            train_matrix=x_train,
            train_labels=y_combined_train,
        )

        no_abstention = classification_metrics(
            y_true=y_bgl_test,
            probabilities=test_probabilities,
            threshold=anomaly_threshold,
        )

        adapted_selective = selective_metrics(
            y_true=y_bgl_test,
            probabilities=test_probabilities,
            maximum_similarity=test_similarity,
            evidence_labels=test_evidence_labels,
            anomaly_threshold=anomaly_threshold,
            policy=safety_policy,
        )

        model_path = (
            ARTIFACT_DIR
            / f"hdfs_bgl_adapted_support_{support_size}.joblib"
        )

        joblib.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "support_size": support_size,
                "anomaly_threshold": anomaly_threshold,
                "ood_cutoff": ood_cutoff,
                "safety_policy": safety_policy,
                "source_domain": "HDFS",
                "target_domain": "BGL",
            },
            model_path,
        )

        all_results["runs"][str(support_size)] = {
            "support_size": support_size,
            "validation_f1": validation_f1,
            "anomaly_threshold": anomaly_threshold,
            "ood_cutoff": ood_cutoff,
            "safety_policy": safety_policy,
            "final_test_no_abstention": no_abstention,
            "final_test_adapted_abstention": adapted_selective,
            "model_path": str(model_path),
        }

        print_result(
            support_size=support_size,
            no_abstention=no_abstention,
            selective=adapted_selective,
            anomaly_threshold=anomaly_threshold,
            policy=safety_policy,
            ood_cutoff=ood_cutoff,
        )

    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved few-shot results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()