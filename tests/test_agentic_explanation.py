import pytest

from app.agentic_explanation import validate_agent_output


def test_agent_output_accepts_only_approved_evidence_ids():
    result = validate_agent_output(
        {
            "explanation": "The known pool incident matches the logs.",
            "cited_evidence_ids": ["INC-DB-001"],
            "uncertainty": "An engineer must still verify the cause.",
        },
        allowed_evidence_ids={"INC-DB-001", "RUNBOOK-DB-01"},
    )

    assert result["cited_evidence_ids"] == ["INC-DB-001"]
    assert result["decision"] == "ENGINEER_REVIEW_REQUIRED"


def test_agent_output_rejects_unknown_evidence_ids():
    with pytest.raises(ValueError, match="outside the approved ledger"):
        validate_agent_output(
            {
                "explanation": "Unsupported claim.",
                "cited_evidence_ids": ["INVENTED-001"],
                "uncertainty": "Uncertain.",
            },
            allowed_evidence_ids={"INC-DB-001"},
        )
