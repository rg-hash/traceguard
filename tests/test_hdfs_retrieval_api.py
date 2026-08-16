import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app import api
from app.retrieval import RetrievedEvidence


class FakeRetriever:
    """Deterministic retriever used to test the API without loading ML models."""

    def search(self, query_text: str, top_k: int = 3):
        assert "blockinfo not found in volumemap" in query_text

        return [
            RetrievedEvidence(
                incident_id="historical-anomaly-1",
                label=1,
                score=0.92,
                text=(
                    "warn dfs.fsdataset: unexpected error trying to delete "
                    "block <block_id>. blockinfo not found in volumemap.\n"
                    "info dfs.fsnamesystem: block* namesystem.delete: "
                    "<block_id> is added to invalidset of <ip_port>"
                ),
            ),
            RetrievedEvidence(
                incident_id="historical-anomaly-2",
                label=1,
                score=0.90,
                text=(
                    "warn dfs.fsdataset: unexpected error trying to delete "
                    "block <block_id>. blockinfo not found in volumemap."
                ),
            ),
            RetrievedEvidence(
                incident_id="historical-anomaly-3",
                label=1,
                score=0.88,
                text=(
                    "warn dfs.fsdataset: unexpected error trying to delete "
                    "block <block_id>. blockinfo not found in volumemap."
                ),
            ),
        ]


def test_hdfs_retrieval_returns_anomaly_with_cited_evidence(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_hdfs_retriever",
        lambda: FakeRetriever(),
    )

    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: None,
    )

    monkeypatch.setattr(
        api,
        "save_incident_decision",
        lambda **kwargs: 123,
    )

    client = TestClient(api.app)

    response = client.post(
        "/retrieve/hdfs",
        json={
            "incident_id": "test-delete-error",
            "events": [
                {
                    "message": (
                        "081111 042720 19510 WARN dfs.FSDataset: "
                        "Unexpected error trying to delete block "
                        "blk_2719230260348020339. "
                        "BlockInfo not found in volumeMap."
                    )
                }
            ],
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert result["decision_record_id"] == 123
    assert result["recommendation"] == "LIKELY_ANOMALY"
    assert result["evidence_labels"] == [1, 1, 1]
    assert len(result["evidence"]) == 3

    citations = result["evidence"][0]["cited_templates"]

    assert len(citations) == 2
    assert citations[0]["score"] == 1.0
    assert "blockinfo not found in volumemap" in citations[0]["template"]