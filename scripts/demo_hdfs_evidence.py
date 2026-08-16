import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evidence import rank_evidence_lines
from app.hybrid_retrieval import HybridRetriever
from app.log_normalization import incident_to_text


DATASET_PATH = ROOT / "data/processed/hdfs_incidents.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/hdfs_hybrid_train_documents.jsonl"

SEMANTIC_WEIGHT = 0.75
MINIMUM_HYBRID_SCORE = 0.05
TOP_K = 3

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

    validation_end = int(len(incidents) * 0.80)
    final_test_incidents = incidents[validation_end:]

    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = HybridRetriever(semantic_weight=SEMANTIC_WEIGHT)
    retriever.build_index(documents)

    # Demo only: select one known anomalous incident from final temporal data.
    query_incident = next(
        incident
        for incident in final_test_incidents
        if incident["is_anomaly"] == 1
    )
    query_text = incident_to_text(query_incident)

    matches = retriever.search(query_text, top_k=TOP_K)

    labels = [match.label for match in matches]
    unanimous = len(set(labels)) == 1
    strong_score = matches[0].score >= MINIMUM_HYBRID_SCORE

    if strong_score and unanimous:
        recommendation = (
            "LIKELY_ANOMALY" if labels[0] == 1 else "LIKELY_NORMAL"
        )
    else:
        recommendation = "NEEDS_HUMAN_REVIEW"

    print(f"Query incident: {query_incident['incident_id']}")
    print("True label: Anomaly")
    print(f"Recommendation: {recommendation}")
    print(f"Top hybrid similarity: {matches[0].score:.4f}")
    print()

    for rank, match in enumerate(matches, start=1):
        label = "Anomaly" if match.label == 1 else "Normal"

        print(f"Evidence {rank}: {label} | score={match.score:.4f}")
        print(f"Historical incident: {match.incident_id}")

        cited_templates = rank_evidence_lines(
            query_text=query_text,
            evidence_text=match.text,
            max_lines=2,
        )

        for citation in cited_templates:
            print(
                f"  - template score={citation['score']}: "
                f"{citation['template']}"
            )

        print()


if __name__ == "__main__":
    main()