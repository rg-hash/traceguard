from app.learning import (
    feedback_to_learning_examples,
    train_feedback_candidate,
)


def test_verified_feedback_trains_review_only_candidate():
    records = []

    for index in range(24):
        label = "ANOMALY" if index % 2 else "NORMAL"
        message = (
            "database timeout connection pool exhausted"
            if label == "ANOMALY"
            else "request completed successfully health check passed"
        )

        records.append(
            {
                "incident_id": f"feedback-{index}",
                "final_anomaly_label": label,
                "incident_events": [{"message": message}],
                "created_at": f"2026-08-{index + 1:02d}T10:00:00Z",
            }
        )

    examples = feedback_to_learning_examples(records)
    _, report = train_feedback_candidate(
        examples,
        minimum_examples_per_class=10,
    )

    assert len(examples) == 24
    assert report["split"] == "chronological 80/20"
    assert report["promotion_status"] == "HUMAN_APPROVAL_REQUIRED"
    assert report["test_f1"] == 1.0
