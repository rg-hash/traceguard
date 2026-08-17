import json
from pathlib import Path


DEVELOPMENT_PATH = Path("data/processed/bgl_development_windows.jsonl")
LOCKED_HOLDOUT_PATH = Path(
    "data/processed/bgl_locked_temporal_holdout.jsonl"
)


def load_incident_numbers(path: Path) -> set[int]:
    with path.open(encoding="utf-8") as file:
        incidents = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    return {
        int(incident["incident_id"].split("-")[-1])
        for incident in incidents
    }


def test_locked_bgl_holdout_is_disjoint_and_temporally_later():
    development_ids = load_incident_numbers(DEVELOPMENT_PATH)
    locked_ids = load_incident_numbers(LOCKED_HOLDOUT_PATH)

    assert development_ids
    assert locked_ids
    assert development_ids.isdisjoint(locked_ids)
    assert max(development_ids) < min(locked_ids)


def test_locked_bgl_holdout_is_balanced():
    with LOCKED_HOLDOUT_PATH.open(encoding="utf-8") as file:
        labels = [
            json.loads(line)["is_anomaly"]
            for line in file
            if line.strip()
        ]

    assert labels.count(0) == 1000
    assert labels.count(1) == 1000