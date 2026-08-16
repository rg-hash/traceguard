import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import SemanticRetriever


DATASET_PATH = ROOT / "data/processed/bgl_windows.jsonl"
INDEX_DIR = ROOT / "data/index"
TRAIN_DOCUMENTS_PATH = INDEX_DIR / "bgl_retrieval_train_documents.jsonl"


def load_bgl_windows() -> list[dict]:
    """Read the BGL windows created by prepare_bgl.py."""
    windows = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                windows.append(json.loads(line))

    return windows


def window_to_text(window: dict) -> str:
    """
    Convert a structured BGL window into one text document.

    We deliberately use log messages only. We do not use `is_anomaly`,
    root-cause labels, or incident IDs as input text.
    """
    messages = [
        str(event["message"])
        for event in window["events"]
        if event.get("message")
    ]
    return "\n".join(messages)


def save_documents(documents: list[dict]) -> None:
    """Save only training documents, preventing test-data leakage."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with TRAIN_DOCUMENTS_PATH.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")

def incident_number(window: dict) -> int:
    """
    Extract the original BGL window number, used to keep the split chronological.
    Example: bgl-window-024856 -> 24856
    """
    return int(str(window["incident_id"]).rsplit("-", maxsplit=1)[1])


def main() -> None:
    windows = load_bgl_windows()

    windows.sort(key=incident_number)

    split_index = int(len(windows) * 0.60)
    train_windows = windows[:split_index]
    future_windows = windows[split_index:]

    documents = [
        {
            "incident_id": str(window["incident_id"]),
            "text": window_to_text(window),
            "label": int(window["is_anomaly"]),
        }
        for window in train_windows
    ]

    save_documents(documents)

    retriever = SemanticRetriever()
    retriever.build_index(documents)
    training_anomalies = sum(window["is_anomaly"] for window in train_windows)
    future_anomalies = sum(window["is_anomaly"] for window in future_windows)
    print(f"Total BGL windows: {len(windows):,}")
    print(f"Training evidence documents: {len(documents):,}")
    print(f"Training anomalies: {training_anomalies:,}")
    print(f"Future holdout anomalies: {future_anomalies:,}")
    print(f"Saved training evidence to: {TRAIN_DOCUMENTS_PATH}")
    print(f"Embedding matrix shape: {retriever.embeddings.shape}")


if __name__ == "__main__":
    main()