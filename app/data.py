"""Safe, reproducible incident data generation and loading."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CAUSES = ("database", "network", "application")
SIGNATURES = {
    "database": ["db connection pool exhausted", "postgres timeout", "query latency elevated"],
    "network": ["packet loss detected", "dns resolution failure", "upstream connection reset"],
    "application": ["null pointer exception", "request handler error", "deployment version mismatch"],
}
SERVICES = ("gateway", "session-manager", "policy-engine", "billing-api")


def generate_incidents(count: int = 240, seed: int = 7) -> list[dict[str, Any]]:
    """Generate labelled windows. Each anomalous window has a causal log signature."""
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, 8, 0, 0)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        anomalous = index % 2 == 0
        cause = CAUSES[index % len(CAUSES)] if anomalous else "normal"
        service = SERVICES[index % len(SERVICES)]
        incident_id = f"demo-{cause}-{index:03d}"
        events = []
        for event_index in range(8):
            timestamp = (start + timedelta(minutes=index * 5, seconds=event_index * 10)).isoformat() + "Z"
            if anomalous and event_index in (3, 4, 5):
                message = SIGNATURES[cause][event_index - 3]
                severity = "ERROR" if event_index != 5 else "WARN"
            else:
                message = rng.choice(["request completed", "health check passed", "cache hit", "session refreshed"])
                severity = "INFO"
            events.append({"timestamp": timestamp, "service": service, "severity": severity, "message": message})
        rows.append({"incident_id": incident_id, "is_anomaly": int(anomalous), "root_cause": cause, "events": events})
    return rows


def write_dataset(path: Path, count: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for incident in generate_incidents(count=count):
            output.write(json.dumps(incident) + "\n")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        write_dataset(path)
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]
