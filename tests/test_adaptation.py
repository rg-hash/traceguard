import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adaptation import select_balanced_support_set


def make_incident(incident_id: str, is_anomaly: int) -> dict:
    return {
        "incident_id": incident_id,
        "is_anomaly": is_anomaly,
        "events": [{"message": "example log"}],
    }


def test_selects_balanced_reproducible_support_set():
    incidents = [
        make_incident("incident-1", 0),
        make_incident("incident-2", 1),
        make_incident("incident-3", 0),
        make_incident("incident-4", 1),
        make_incident("incident-5", 0),
        make_incident("incident-6", 1),
    ]

    first_selection = select_balanced_support_set(
        incidents,
        total_size=4,
        seed=7,
    )
    second_selection = select_balanced_support_set(
        incidents,
        total_size=4,
        seed=7,
    )

    assert len(first_selection) == 4
    assert sum(item["is_anomaly"] == 0 for item in first_selection) == 2
    assert sum(item["is_anomaly"] == 1 for item in first_selection) == 2

    first_ids = [item["incident_id"] for item in first_selection]
    second_ids = [item["incident_id"] for item in second_selection]

    assert first_ids == second_ids


def test_rejects_odd_support_size():
    incidents = [
        make_incident("incident-1", 0),
        make_incident("incident-2", 1),
    ]

    try:
        select_balanced_support_set(incidents, total_size=3)
    except ValueError as error:
        assert "even" in str(error)
    else:
        raise AssertionError("Expected ValueError for odd support size.")