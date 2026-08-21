import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from app import api
from app.investigation import get_investigation_service


def test_database_logs_create_grounded_investigation_plan():
    service = get_investigation_service()

    result = service.investigate(
        incident_id="payment-incident-001",
        triage_recommendation="LIKELY_ANOMALY",
        triage_context={
            "source": "bgl_evidence_grounded_triage",
            "classifier_anomaly_probability": 0.91,
            "classifier_prediction": "Anomaly",
            "decision_checks": {
                "classifier_confidence": 0.91,
                "top_similarity": 0.88,
                "classifier_evidence_agree": True,
            },
            "evidence": [
                {
                    "historical_incident_id": "bgl-window-000123",
                    "historical_label": "Anomaly",
                    "similarity": 0.88,
                }
            ],
        },
        events=[
            {
                "timestamp": "2026-08-20T10:08:00Z",
                "service": "payment-api",
                "severity": "ERROR",
                "message": (
                    "database connection timeout; "
                    "connection pool exhausted"
                ),
            }
        ],
    )

    assert result["decision"] == "ENGINEER_REVIEW_REQUIRED"

    assert result["triage_context"][
        "classifier_anomaly_probability"
    ] == 0.91

    assert result["summary"]["affected_service"] == "payment-api"

    assert result["hypotheses"][0]["cause"] == (
        "database_connection_pool_exhaustion"
    )

    assert "INC-DB-001" in result["hypotheses"][0][
        "evidence_ids"
    ]

    assert result["recommended_checks"]

    assert any(
        check["source_id"] == "RUNBOOK-DB-01"
        for check in result["recommended_checks"]
    )


class FakeInvestigationService:
    """Avoid real ML/database startup while testing the API contract."""

    def investigate(
        self,
        incident_id: str,
        events: list[dict],
        triage_recommendation: str,
        triage_context: dict,
    ) -> dict:
        return {
            "incident_id": incident_id,
            "triage_recommendation": triage_recommendation,
            "triage_context": triage_context,
            "summary": {
                "affected_service": "checkout-api",
                "event_count": len(events),
            },
            "evidence": [],
            "deployment_context": [],
            "hypotheses": [],
            "recommended_checks": [],
            "decision": "ENGINEER_REVIEW_REQUIRED",
        }


def test_investigation_api_preserves_bgl_triage_context(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: None,
    )

    monkeypatch.setattr(
        api,
        "get_hdfs_retriever",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_bgl_triage_service",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_investigation_service",
        lambda: FakeInvestigationService(),
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/investigate",
            json={
                "incident_id": "network-incident-001",
                "triage_recommendation": (
                    "NEEDS_HUMAN_REVIEW"
                ),
                "triage_context": {
                    "source": (
                        "bgl_evidence_grounded_triage"
                    ),
                    "classifier_anomaly_probability": 0.54,
                    "classifier_prediction": "Anomaly",
                    "decision_checks": {
                        "classifier_confidence": 0.54,
                        "top_similarity": 0.36,
                        "classifier_evidence_agree": False,
                    },
                    "evidence": [
                        {
                            "historical_incident_id": (
                                "bgl-window-000987"
                            ),
                            "historical_label": "Normal",
                            "similarity": 0.36,
                            "cited_templates": [],
                        }
                    ],
                },
                "events": [
                    {
                        "service": "checkout-api",
                        "severity": "ERROR",
                        "message": (
                            "dns resolution failure; "
                            "upstream connection reset"
                        ),
                    }
                ],
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["decision"] == "ENGINEER_REVIEW_REQUIRED"

    assert result["triage_recommendation"] == (
        "NEEDS_HUMAN_REVIEW"
    )

    assert result["triage_context"][
        "classifier_anomaly_probability"
    ] == 0.54

    assert result["triage_context"]["evidence"][0][
        "historical_incident_id"
    ] == "bgl-window-000987"

def test_feedback_api_stores_human_label(monkeypatch):
    saved = {}

    def fake_save_investigation_feedback(**kwargs):
        saved.update(kwargs)
        return 101

    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: None,
    )

    monkeypatch.setattr(
        api,
        "get_hdfs_retriever",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_bgl_triage_service",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_investigation_service",
        lambda: FakeInvestigationService(),
    )

    monkeypatch.setattr(
        api,
        "save_investigation_feedback",
        fake_save_investigation_feedback,
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/investigations/feedback",
            json={
                "incident_id": "payment-incident-001",
                "triage_recommendation": "LIKELY_ANOMALY",
                "hypothesis": (
                    "database_connection_pool_exhaustion"
                ),
                "hypothesis_accepted": True,
                "confirmed_root_cause": (
                    "database_connection_pool_exhaustion"
                ),
                "resolution": (
                    "Fixed the connection leak and "
                    "increased pool capacity."
                ),
                "usefulness_rating": 5,
                "reviewer_note": (
                    "Suggested checks were useful."
                ),
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "feedback_id": 101,
        "status": "stored",
    }

    assert saved["incident_id"] == "payment-incident-001"

    assert saved["hypothesis_accepted"] is True

    assert saved["usefulness_rating"] == 5


def test_feedback_api_returns_persisted_records(monkeypatch):
    expected = [
        {
            "id": 101,
            "incident_id": "payment-incident-001",
            "triage_recommendation": "LIKELY_ANOMALY",
            "hypothesis": (
                "database_connection_pool_exhaustion"
            ),
            "hypothesis_accepted": True,
            "confirmed_root_cause": (
                "database_connection_pool_exhaustion"
            ),
            "resolution": "Fixed connection leak.",
            "usefulness_rating": 5,
            "reviewer_note": "Useful plan.",
            "created_at": "2026-08-21T10:00:00Z",
        }
    ]

    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: None,
    )

    monkeypatch.setattr(
        api,
        "get_hdfs_retriever",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_bgl_triage_service",
        lambda: object(),
    )

    monkeypatch.setattr(
        api,
        "get_investigation_service",
        lambda: FakeInvestigationService(),
    )

    monkeypatch.setattr(
        api,
        "list_investigation_feedback",
        lambda incident_id, limit: expected,
    )

    with TestClient(api.app) as client:
        response = client.get(
            "/investigations/feedback?limit=20"
        )

    assert response.status_code == 200

    assert response.json() == expected


def test_organization_onboarding_stores_isolated_context(monkeypatch):
    saved = {}

    def fake_save_organization_profile(**kwargs):
        saved.update(kwargs)
        return 1

    monkeypatch.setattr(api, "initialize_database", lambda: None)
    monkeypatch.setattr(api, "get_hdfs_retriever", lambda: object())
    monkeypatch.setattr(
        api, "get_bgl_triage_service", lambda: object()
    )
    monkeypatch.setattr(
        api,
        "get_investigation_service",
        lambda: FakeInvestigationService(),
    )
    monkeypatch.setattr(
        api,
        "save_organization_profile",
        fake_save_organization_profile,
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/organizations/onboard",
            json={
                "organization_id": "acme-shop",
                "display_name": "Acme Shop",
                "description": "E-commerce platform.",
                "services": [
                    {
                        "name": "checkout-api",
                        "description": "Creates customer orders.",
                        "dependencies": ["payment-api"],
                    }
                ],
                "knowledge": [
                    {
                        "id": "ACME-RB-01",
                        "kind": "runbook",
                        "title": "Checkout dependency timeout",
                        "service": "checkout-api",
                        "symptoms": ["upstream timeout"],
                        "steps": ["Check payment-api latency."],
                        "tags": ["checkout", "timeout"],
                    }
                ],
                "deployments": [
                    {
                        "id": "acme-deploy-1",
                        "service": "checkout-api",
                        "commit": "abc123",
                        "timestamp": "2026-08-21T10:00:00Z",
                        "summary": "Checkout update.",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "organization_id": "acme-shop",
        "knowledge_version": 1,
        "status": "onboarded",
    }
    assert saved["services"][0]["name"] == "checkout-api"
    assert saved["knowledge"][0]["id"] == "ACME-RB-01"


def test_organization_investigation_uses_versioned_profile(
    monkeypatch,
):
    requested = {}

    monkeypatch.setattr(api, "initialize_database", lambda: None)
    monkeypatch.setattr(api, "get_hdfs_retriever", lambda: object())
    monkeypatch.setattr(
        api, "get_bgl_triage_service", lambda: object()
    )
    monkeypatch.setattr(
        api,
        "get_investigation_service",
        lambda: FakeInvestigationService(),
    )
    monkeypatch.setattr(
        api,
        "get_organization_profile",
        lambda _: {"knowledge_version": 4},
    )

    def fake_organization_service(org_id, version):
        requested["organization_id"] = org_id
        requested["knowledge_version"] = version
        return FakeInvestigationService()

    monkeypatch.setattr(
        api,
        "get_organization_investigation_service",
        fake_organization_service,
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/investigate",
            json={
                "organization_id": "acme-shop",
                "incident_id": "acme-incident-1",
                "events": [
                    {
                        "service": "checkout-api",
                        "severity": "ERROR",
                        "message": "upstream timeout",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert requested == {
        "organization_id": "acme-shop",
        "knowledge_version": 4,
    }
    assert response.json()["organization_id"] == "acme-shop"
