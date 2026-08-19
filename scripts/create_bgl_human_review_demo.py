import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.bgl_triage import (
    BGL_DEVELOPMENT_PATH,
    bgl_time_key,
    get_bgl_triage_service,
    load_jsonl,
)


OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts/demo/bgl_human_review_request.json"
)


def main() -> None:
    """
    Find a real development-era BGL incident for which the deployed
    safety policy chooses NEEDS_HUMAN_REVIEW.

    Its true offline label is deliberately not used as API input.
    """

    service = get_bgl_triage_service()

    development_incidents = load_jsonl(BGL_DEVELOPMENT_PATH)
    development_incidents.sort(key=bgl_time_key)

    validation_start = int(len(development_incidents) * 0.60)
    validation_end = int(len(development_incidents) * 0.80)

    candidate_incidents = development_incidents[
        validation_start:validation_end
    ]

    for candidate in candidate_incidents:
        result = service.triage(
            incident_id="dashboard-human-review-demo",
            events=candidate["events"],
        )

        if result["recommendation"] != "NEEDS_HUMAN_REVIEW":
            continue

        request_payload = {
            "incident_id": "dashboard-human-review-demo",
            "events": candidate["events"],
        }

        output_path = __import__("pathlib").Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(request_payload, indent=2),
            encoding="utf-8",
        )

        print("Created a human-review demo request:")
        print(output_path)
        print()
        print("Safety checks:")
        print(json.dumps(result["decision_checks"], indent=2))
        print()
        print("Top evidence labels:", result["evidence_labels"])

        return

    raise RuntimeError(
        "No abstained incident was found in the development validation "
        "period. Check the deployed safety policy."
    )


if __name__ == "__main__":
    main()