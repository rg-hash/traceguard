import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import SemanticRetriever


DATASET_PATH = ROOT / "data/processed/bgl_windows.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/bgl_retrieval_train_documents.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def window_to_text(window: dict) -> str:
    return "\n".join(
        str(event["message"])
        for event in window["events"]
        if event.get("message")
    )

def incident_number(window: dict) -> int:
    return int(str(window["incident_id"]).rsplit("-", maxsplit=1)[1])

def main() -> None:
    all_windows = load_jsonl(DATASET_PATH)
    all_windows.sort(key=incident_number)

    split_index = int(len(all_windows) * 0.60)
    test_windows = all_windows[split_index:]

    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = SemanticRetriever()
    retriever.build_index(documents)

    # Select one anomalous window from the held-out 40% test partition.
    query_window = next(
        window for window in test_windows if window["is_anomaly"]
    )
    query_text = window_to_text(query_window)

    result = retriever.evidence_summary(
        query_text=query_text,
        top_k=3,
        minimum_similarity=0.55,
    )

    print(f"Query incident ID: {query_window['incident_id']}")
    print(f"True label: {'Anomaly' if query_window['is_anomaly'] else 'Normal'}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Top similarity: {result['top_similarity']}")
    print(f"Anomaly vote ratio: {result['anomaly_vote_ratio']}")
    print()

    print("Retrieved training evidence:")
    for rank, match in enumerate(result["matches"], start=1):
        label = "Anomaly" if match.label == 1 else "Normal"
        preview = match.text.replace("\n", " ")[:180]

        print(f"{rank}. {label} | similarity={match.score:.4f}")
        print(f"   Incident: {match.incident_id}")
        print(f"   Evidence: {preview}...")


if __name__ == "__main__":
    main()