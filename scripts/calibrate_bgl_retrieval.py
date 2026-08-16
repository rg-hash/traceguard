import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import SemanticRetriever


DATASET_PATH = ROOT / "data/processed/bgl_windows.jsonl"
TRAIN_DOCUMENTS_PATH = ROOT / "data/index/bgl_retrieval_train_documents.jsonl"

TOP_K = 3
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]


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


def evaluate_policy(
    all_matches: list,
    true_labels: list[int],
    threshold: float,
    require_unanimous_votes: bool,
) -> tuple[float, float, float]:
    """
    Returns coverage, accuracy among automated decisions,
    and unsafe-decision rate.
    """
    automated = 0
    correct = 0
    unsafe = 0

    for matches, true_label in zip(all_matches, true_labels):
        top_score = matches[0].score
        labels = [match.label for match in matches]

        strong_similarity = top_score >= threshold
        unanimous = len(set(labels)) == 1

        if not strong_similarity:
            continue

        if require_unanimous_votes and not unanimous:
            continue

        predicted_label = int(sum(labels) > len(labels) / 2)

        automated += 1

        if predicted_label == true_label:
            correct += 1
        else:
            unsafe += 1

    coverage = automated / len(true_labels)
    accuracy = correct / automated if automated else 0.0
    unsafe_rate = unsafe / automated if automated else 0.0

    return coverage, accuracy, unsafe_rate


def main() -> None:
    windows = load_jsonl(DATASET_PATH)
    windows.sort(key=incident_number)

    # 60% historic data: retrieval evidence corpus
    # next 20%: validation data for policy selection
    # final 20%: untouched test data for later reporting
    train_end = int(len(windows) * 0.60)
    validation_end = int(len(windows) * 0.80)

    validation_windows = windows[train_end:validation_end]

    documents = load_jsonl(TRAIN_DOCUMENTS_PATH)

    retriever = SemanticRetriever()
    retriever.build_index(documents)

    validation_texts = [window_to_text(window) for window in validation_windows]
    validation_labels = [
        int(window["is_anomaly"]) for window in validation_windows
    ]

    all_matches = retriever.search_batch(validation_texts, top_k=TOP_K)

    print("BGL Validation Calibration")
    print(f"Validation windows: {len(validation_windows):,}")
    print()
    print("Policy       Threshold  Coverage  Accuracy  Unsafe rate")
    print("-" * 58)

    for threshold in THRESHOLDS:
        for name, unanimous in [
            ("majority", False),
            ("unanimous", True),
        ]:
            coverage, accuracy, unsafe_rate = evaluate_policy(
                all_matches=all_matches,
                true_labels=validation_labels,
                threshold=threshold,
                require_unanimous_votes=unanimous,
            )

            print(
                f"{name:10}   {threshold:>6.2f}   "
                f"{coverage:>7.3f}   {accuracy:>7.3f}   {unsafe_rate:>9.3f}"
            )


if __name__ == "__main__":
    main()