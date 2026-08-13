from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_incident_is_404():
    assert client.get("/incidents/no-such-incident/analyze").status_code == 404
