import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hybrid_retrieval import HybridRetriever
from app.log_normalization import incident_to_text


DATASET_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/hdfs_hybrid_train_documents.jsonl"

TOP_K = 3
SEMANTIC_WEIGHT = 0.75
MINIMUM_HYBRID_SCORE = 0.05

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


def main() -> None:
    incidents = load_jsonl(DATASET_PATH)
    incidents.sort(key=incident_time_key)

    train_end = int(len(incidents) * 0.60)
    validation_end = int(len(incidents) * 0.80)

    # These 400 incidents were not used for indexing or policy selection.
    test_incidents = incidents[validation_end:]

    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = HybridRetriever(semantic_weight=SEMANTIC_WEIGHT)
    retriever.build_index(documents)

    test_texts = [incident_to_text(incident) for incident in test_incidents]
    true_labels = np.array(
        [int(incident["is_anomaly"]) for incident in test_incidents]
    )
    document_labels = np.array(
        [int(document["label"]) for document in documents]
    )

    start_time = time.perf_counter()
    semantic_scores, lexical_scores = retriever.score_batch(test_texts)
    elapsed_seconds = time.perf_counter() - start_time

    hybrid_scores = (
        SEMANTIC_WEIGHT * semantic_scores
        + (1.0 - SEMANTIC_WEIGHT) * lexical_scores
    )

    top_indices = np.argsort(hybrid_scores, axis=1)[:, ::-1][:, :TOP_K]
    top_scores = hybrid_scores[
        np.arange(len(test_incidents)),
        top_indices[:, 0],
    ]
    top_labels = document_labels[top_indices]

    precision_at_1 = (top_labels[:, 0] == true_labels).mean()
    precision_at_3 = (top_labels == true_labels[:, None]).mean()

    unanimous = np.all(top_labels == top_labels[:, [0]], axis=1)
    automated = (top_scores >= MINIMUM_HYBRID_SCORE) & unanimous
    predictions = top_labels[:, 0]

    coverage = automated.mean()
    automated_accuracy = (
        (predictions[automated] == true_labels[automated]).mean()
        if automated.any()
        else 0.0
    )
    unsafe_count = int(
        (predictions[automated] != true_labels[automated]).sum()
    )
    unsafe_rate = 1.0 - automated_accuracy if automated.any() else 0.0
    latency_ms = elapsed_seconds * 1000 / len(test_incidents)

    print("HDFS Final Temporal-Test Hybrid Retrieval Evaluation")
    print("-" * 52)
    print(f"Training evidence incidents: {len(documents):,}")
    print(f"Final test incidents: {len(test_incidents):,}")
    print(
        "Selected policy: "
        f"{SEMANTIC_WEIGHT:.2f} semantic / "
        f"{1.0 - SEMANTIC_WEIGHT:.2f} lexical, "
        "unanimous top-3 evidence"
    )
    print(f"Precision@1 (same-label evidence): {precision_at_1:.4f}")
    print(f"Precision@3 (same-label evidence): {precision_at_3:.4f}")
    print(f"Abstention coverage: {coverage:.4f}")
    print(f"Automated-decision accuracy: {automated_accuracy:.4f}")
    print(f"Unsafe confident decision rate: {unsafe_rate:.4f}")
    print(f"Unsafe confident decisions: {unsafe_count:,}")
    print(f"Hybrid retrieval latency: {latency_ms:.2f} ms/query")


if __name__ == "__main__":
    main()