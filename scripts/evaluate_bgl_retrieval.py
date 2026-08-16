import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import SemanticRetriever


DATASET_PATH = ROOT / "data/processed/bgl_windows.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/bgl_retrieval_train_documents.jsonl"

TOP_K = 3
SELECTED_SIMILARITY_THRESHOLD = 0.90
REQUIRE_UNANIMOUS_VOTES = True


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def incident_number(window: dict) -> int:
    return int(str(window["incident_id"]).rsplit("-", maxsplit=1)[1])


def window_to_text(window: dict) -> str:
    return "\n".join(
        str(event["message"])
        for event in window["events"]
        if event.get("message")
    )


def main() -> None:
    windows = load_jsonl(DATASET_PATH)
    windows.sort(key=incident_number)

    train_end = int(len(windows) * 0.60)
    validation_end = int(len(windows) * 0.80)

    # This final 20% was not used to build the index or choose the policy.
    test_windows = windows[validation_end:]

    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = SemanticRetriever()
    retriever.build_index(documents)

    test_texts = [window_to_text(window) for window in test_windows]
    true_labels = [int(window["is_anomaly"]) for window in test_windows]

    start_time = time.perf_counter()
    all_matches = retriever.search_batch(test_texts, top_k=TOP_K)
    elapsed_seconds = time.perf_counter() - start_time

    correct_at_1 = 0
    matching_labels_at_3 = 0
    automated = 0
    correct_automated = 0
    unsafe_automated = 0

    for true_label, matches in zip(true_labels, all_matches):
        labels = [match.label for match in matches]

        correct_at_1 += int(labels[0] == true_label)
        matching_labels_at_3 += sum(label == true_label for label in labels)

        strong_similarity = (
            matches[0].score >= SELECTED_SIMILARITY_THRESHOLD
        )
        unanimous = len(set(labels)) == 1

        if not strong_similarity:
            continue

        if REQUIRE_UNANIMOUS_VOTES and not unanimous:
            continue

        predicted_label = labels[0]
        automated += 1

        if predicted_label == true_label:
            correct_automated += 1
        else:
            unsafe_automated += 1

    total = len(test_windows)
    precision_at_1 = correct_at_1 / total
    precision_at_3 = matching_labels_at_3 / (total * TOP_K)
    coverage = automated / total
    automated_accuracy = correct_automated / automated if automated else 0.0
    unsafe_rate = unsafe_automated / automated if automated else 0.0
    latency_ms = elapsed_seconds * 1000 / total

    print("BGL Final Temporal-Test Evaluation")
    print("-" * 40)
    print(f"Training evidence windows: {len(documents):,}")
    print(f"Final test queries: {total:,}")
    print(
        "Selected policy: unanimous top-3 labels and "
        f"similarity >= {SELECTED_SIMILARITY_THRESHOLD:.2f}"
    )
    print(f"Precision@1 (same-label evidence): {precision_at_1:.4f}")
    print(f"Precision@3 (same-label evidence): {precision_at_3:.4f}")
    print(f"Abstention coverage: {coverage:.4f}")
    print(f"Automated-decision accuracy: {automated_accuracy:.4f}")
    print(f"Unsafe confident decision rate: {unsafe_rate:.4f}")
    print(f"Unsafe confident decisions: {unsafe_automated:,}")
    print(f"Retrieval latency per query: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()