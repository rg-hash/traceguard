"""Controlled offline learning from human-verified TraceGuard feedback."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


LABEL_TO_INT = {
    "NORMAL": 0,
    "ANOMALY": 1,
}


def feedback_to_learning_examples(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert verified feedback into chronological text-classification rows."""
    examples = []

    for record in records:
        label_name = record.get("final_anomaly_label")

        if label_name not in LABEL_TO_INT:
            continue

        messages = [
            str(event.get("message", "")).strip()
            for event in record.get("incident_events", [])
            if str(event.get("message", "")).strip()
        ]

        if not messages:
            continue

        examples.append(
            {
                "incident_id": str(record["incident_id"]),
                "text": " ".join(messages),
                "label": LABEL_TO_INT[label_name],
                "created_at": str(record.get("created_at", "")),
            }
        )

    return sorted(
        examples,
        key=lambda example: example["created_at"],
    )


def train_feedback_candidate(
    examples: list[dict[str, Any]],
    *,
    minimum_examples_per_class: int = 10,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train and evaluate a candidate with a chronological 80/20 split.

    This function intentionally does not deploy the returned model. A human
    must review its report before any separate promotion operation.
    """
    counts = Counter(example["label"] for example in examples)

    if min(counts.get(0, 0), counts.get(1, 0)) < minimum_examples_per_class:
        raise ValueError(
            "At least "
            f"{minimum_examples_per_class} verified NORMAL and ANOMALY "
            "examples are required for candidate training."
        )

    split_index = int(len(examples) * 0.80)
    train_examples = examples[:split_index]
    test_examples = examples[split_index:]

    train_counts = Counter(
        example["label"] for example in train_examples
    )
    test_counts = Counter(
        example["label"] for example in test_examples
    )

    if (
        min(train_counts.get(0, 0), train_counts.get(1, 0)) == 0
        or min(test_counts.get(0, 0), test_counts.get(1, 0)) == 0
    ):
        raise ValueError(
            "Chronological split must contain both classes in train and test. "
            "Collect more verified feedback before training."
        )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=7,
                ),
            ),
        ]
    )

    model.fit(
        [example["text"] for example in train_examples],
        [example["label"] for example in train_examples],
    )

    predictions = model.predict(
        [example["text"] for example in test_examples]
    )
    labels = [example["label"] for example in test_examples]

    report = {
        "method": "TF-IDF (1,2-grams) + Logistic Regression",
        "split": "chronological 80/20",
        "training_examples": len(train_examples),
        "test_examples": len(test_examples),
        "training_class_counts": dict(train_counts),
        "test_class_counts": dict(test_counts),
        "test_f1": round(
            float(f1_score(labels, predictions, zero_division=0)),
            4,
        ),
        "test_precision": round(
            float(
                precision_score(labels, predictions, zero_division=0)
            ),
            4,
        ),
        "test_recall": round(
            float(recall_score(labels, predictions, zero_division=0)),
            4,
        ),
        "promotion_status": "HUMAN_APPROVAL_REQUIRED",
    }

    return model, report
