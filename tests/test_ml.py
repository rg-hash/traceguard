from app.data import generate_incidents
from app.ml import analyze, train_models


def test_network_incident_returns_cited_diagnosis():
    rows = generate_incidents(60)
    models = train_models(rows)
    incident = next(row for row in rows if row["root_cause"] == "network")
    result = analyze(incident, models)
    assert result["decision"] == "ANALYZED"
    assert result["root_cause"] == "network"
    assert result["evidence"]


def test_normal_event_abstains():
    rows = generate_incidents(60)
    models = train_models(rows)
    normal = next(row for row in rows if row["root_cause"] == "normal")
    result = analyze(normal, models)
    assert result["decision"] == "NEEDS_HUMAN_REVIEW"
    assert result["root_cause"] is None
