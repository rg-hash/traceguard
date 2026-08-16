import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hybrid_retrieval import HybridRetriever
from app.log_normalization import incident_to_text


DATASET_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
INDEX_DIR = ROOT / "data/index"
TRAIN_DOCUMENTS_PATH = INDEX_DIR / "hdfs_hybrid_train_documents.jsonl"

HDFS_TIMESTAMP = re.compile(r"^(\d{6})\s+(\d{6})\b")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def incident_time_key(incident: dict) -> str:
    """
    Derive a sortable timestamp from the first HDFS log event.

    HDFS logs start with a date and time such as:
    081111 042720
    """
    for event in incident.get("events", []):
        message = str(event.get("message", ""))

        match = HDFS_TIMESTAMP.match(message)
        if match:
            return match.group(1) + match.group(2)

    raise ValueError(
        f"No timestamp found for incident {incident['incident_id']}"
    )


def save_jsonl(path: Path, documents: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")


def main() -> None:
    incidents = load_jsonl(DATASET_PATH)
    incidents.sort(key=incident_time_key)

    train_end = int(len(incidents) * 0.60)
    validation_end = int(len(incidents) * 0.80)

    train_incidents = incidents[:train_end]
    validation_incidents = incidents[train_end:validation_end]
    test_incidents = incidents[validation_end:]

    documents = [
        {
            "incident_id": str(incident["incident_id"]),
            "text": incident_to_text(incident),
            "label": int(incident["is_anomaly"]),
        }
        for incident in train_incidents
    ]

    save_jsonl(TRAIN_DOCUMENTS_PATH, documents)

    retriever = HybridRetriever(semantic_weight=0.60)
    retriever.build_index(documents)

    print(f"Total HDFS incidents: {len(incidents):,}")
    print(
        "Split sizes "
        f"(train / validation / test): "
        f"{len(train_incidents):,} / "
        f"{len(validation_incidents):,} / "
        f"{len(test_incidents):,}"
    )
    print(
        "Train anomalies: "
        f"{sum(item['is_anomaly'] for item in train_incidents):,}"
    )
    print(
        "Validation anomalies: "
        f"{sum(item['is_anomaly'] for item in validation_incidents):,}"
    )
    print(
        "Test anomalies: "
        f"{sum(item['is_anomaly'] for item in test_incidents):,}"
    )
    print(f"Saved training evidence: {TRAIN_DOCUMENTS_PATH}")
    print(
        "Semantic embedding shape: "
        f"{retriever.semantic_embeddings.shape}"
    )
    print(f"TF-IDF matrix shape: {retriever.tfidf_matrix.shape}")


if __name__ == "__main__":
    main()