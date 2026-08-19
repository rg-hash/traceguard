import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import api


class FakeBGLTriageService:
    """Small deterministic replacement for the real ML service in a unit test."""

    def triage(self, incident_id: str, events: list[dict]) -> dict:
        return {
            "incident_id": incident_id,
            "recommendation": "LIKELY_ANOMALY",
            "classifier_anomaly_probability": 0.91,
            "classifier_prediction": "Anomaly",
            "policy": {
                "model": "TF-IDF + Logistic Regression",
                "support_size": 600,
                "requires_unanimous_top_3_evidence": True,
                "requires_classifier_evidence_agreement": True,
            },
            "decision_checks": {
                "classifier_confidence": 0.91,
                "top_similarity": 0.88,
                "confidence_passed": True,
                "similarity_passed": True,
                "top_3_evidence_unanimous": True,
                "classifier_evidence_agree": True,
            },
            "evidence_labels": [1, 1, 1],
            "evidence": [
                {
                    "rank": 1,
                    "historical_incident_id": "bgl-window-000123",
                    "historical_label": "Anomaly",
                    "similarity": 0.88,
                    "cited_templates": [
                        {
                            "template": (
                                "ras kernel fatal data storage interrupt"
                            ),
                            "score": 1.0,
                        }
                    ],
                }
            ],
        }


def test_bgl_triage_returns_evidence_and_persists_decision(
    monkeypatch,
):
    saved = {}

    def fake_save_incident_decision(**kwargs):
        saved.update(kwargs)
        return 456

    # Prevent startup from loading real models or connecting to PostgreSQL.
    monkeypatch.setattr(api, "initialize_database", lambda: None)
    monkeypatch.setattr(api, "get_hdfs_retriever", lambda: object())
    monkeypatch.setattr(
        api,
        "get_bgl_triage_service",
        lambda: FakeBGLTriageService(),
    )
    monkeypatch.setattr(
        api,
        "save_incident_decision",
        fake_save_incident_decision,
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/triage/bgl",
            json={
                "incident_id": "test-bgl-fatal-001",
                "events": [
                    {
                        "message": (
                            "RAS KERNEL FATAL "
                            "data storage interrupt"
                        )
                    }
                ],
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["recommendation"] == "LIKELY_ANOMALY"
    assert result["decision_record_id"] == 456
    assert result["evidence_labels"] == [1, 1, 1]
    assert result["evidence"][0]["historical_label"] == "Anomaly"

    assert saved["incident_id"] == "test-bgl-fatal-001"
    assert saved["source"] == "bgl_evidence_grounded_triage"
    assert saved["recommendation"] == "LIKELY_ANOMALY"
    assert saved["top_similarity"] == 0.88