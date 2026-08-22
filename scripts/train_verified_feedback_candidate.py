"""Train an offline candidate from human-verified organization feedback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import list_verified_learning_feedback
from app.learning import (
    feedback_to_learning_examples,
    train_feedback_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a review-only anomaly candidate from verified feedback."
        )
    )
    parser.add_argument("--organization-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/feedback_candidates",
    )
    arguments = parser.parse_args()

    records = list_verified_learning_feedback(
        organization_id=arguments.organization_id,
    )
    examples = feedback_to_learning_examples(records)
    model, report = train_feedback_candidate(examples)

    output_dir = PROJECT_ROOT / arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = (
        output_dir
        / f"{arguments.organization_id}_candidate.joblib"
    )
    report_path = (
        output_dir
        / f"{arguments.organization_id}_candidate_report.json"
    )

    joblib.dump(
        {
            "model": model,
            "organization_id": arguments.organization_id,
            "report": report,
        },
        artifact_path,
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Candidate artifact: {artifact_path}")
    print(f"Evaluation report: {report_path}")
    print(
        "Promotion status: HUMAN_APPROVAL_REQUIRED "
        "(candidate was not deployed)."
    )


if __name__ == "__main__":
    main()
