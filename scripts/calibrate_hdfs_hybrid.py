import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hybrid_retrieval import HybridRetriever
from app.log_normalization import incident_to_text


DATASET_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/hdfs_hybrid_train_documents.jsonl"

TOP_K = 3
SEMANTIC_WEIGHTS = [0.00, 0.25, 0.50, 0.60, 0.75, 1.00]
THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]
MAX_UNSAFE_RATE = 0.05

HDFS_TIMESTAMP = re.compile(r"^(\d{6})\s+(\d{6})\b")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def incident_time_key(incident: dict) -> str:
    for event in incident.get("events", []):
        match = HDFS_TIMESTAMP.match(str(event.get("message", "")))
        if match:
            return match.group(1) + match.group(2)

    raise ValueError(f"No timestamp for {incident['incident_id']}")


def evaluate_policy(
    scores: np.ndarray,
    document_labels: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Evaluate a cautious policy:
    - top hybrid score must exceed threshold
    - all top-3 evidence labels must agree
    """
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = scores[
        np.arange(len(scores)),
        top_indices[:, 0],
    ]
    top_labels = document_labels[top_indices]

    unanimous = np.all(top_labels == top_labels[:, [0]], axis=1)
    automated = (top_scores >= threshold) & unanimous
    predictions = top_labels[:, 0]

    coverage = automated.mean()

    if not automated.any():
        return coverage, 0.0, 0.0

    accuracy = (predictions[automated] == true_labels[automated]).mean()
    unsafe_rate = 1.0 - accuracy

    return coverage, accuracy, unsafe_rate


def ranking_metrics(
    scores: np.ndarray,
    document_labels: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[float, float]:
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :TOP_K]
    top_labels = document_labels[top_indices]

    precision_at_1 = (top_labels[:, 0] == true_labels).mean()
    precision_at_3 = (top_labels == true_labels[:, None]).mean()

    return precision_at_1, precision_at_3


def main() -> None:
    incidents = load_jsonl(DATASET_PATH)
    incidents.sort(key=incident_time_key)

    train_end = int(len(incidents) * 0.60)
    validation_end = int(len(incidents) * 0.80)

    validation_incidents = incidents[train_end:validation_end]
    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = HybridRetriever(semantic_weight=0.60)
    retriever.build_index(documents)

    validation_texts = [
        incident_to_text(incident) for incident in validation_incidents
    ]
    true_labels = np.array(
        [int(incident["is_anomaly"]) for incident in validation_incidents]
    )
    document_labels = np.array(
        [int(document["label"]) for document in documents]
    )

    semantic_scores, lexical_scores = retriever.score_batch(validation_texts)

    print("HDFS Hybrid Retrieval Validation Calibration")
    print(f"Validation queries: {len(validation_incidents):,}")
    print("Safety rule: unanimous top-3 evidence, unsafe rate <= 5%")
    print()
    print(
        "Semantic  P@1    P@3    Selected threshold  "
        "Coverage  Accuracy  Unsafe rate"
    )
    print("-" * 78)

    for semantic_weight in SEMANTIC_WEIGHTS:
        hybrid_scores = (
            semantic_weight * semantic_scores
            + (1.0 - semantic_weight) * lexical_scores
        )

        precision_at_1, precision_at_3 = ranking_metrics(
            scores=hybrid_scores,
            document_labels=document_labels,
            true_labels=true_labels,
        )

        best_policy = None

        for threshold in THRESHOLDS:
            coverage, accuracy, unsafe_rate = evaluate_policy(
                scores=hybrid_scores,
                document_labels=document_labels,
                true_labels=true_labels,
                threshold=threshold,
            )

            if unsafe_rate <= MAX_UNSAFE_RATE:
                if best_policy is None or coverage > best_policy[1]:
                    best_policy = (
                        threshold,
                        coverage,
                        accuracy,
                        unsafe_rate,
                    )

        if best_policy is None:
            print(
                f"{semantic_weight:>7.2f}   "
                f"{precision_at_1:>5.3f}  {precision_at_3:>5.3f}  "
                "No policy met the 5% safety target"
            )
            continue

        threshold, coverage, accuracy, unsafe_rate = best_policy

        print(
            f"{semantic_weight:>7.2f}   "
            f"{precision_at_1:>5.3f}  {precision_at_3:>5.3f}  "
            f"{threshold:>18.2f}  "
            f"{coverage:>7.3f}  {accuracy:>7.3f}  {unsafe_rate:>9.3f}"
        )


if __name__ == "__main__":
    main()