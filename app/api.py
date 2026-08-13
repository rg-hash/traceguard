from __future__ import annotations

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import DATASET_PATH, MODEL_PATH
from app.data import load_dataset
from app.ml import analyze, train_models

app = FastAPI(title="TraceGuard", version="1.0.0", description="Evidence-grounded AIOps incident analysis.")


class LogEvent(BaseModel):
    timestamp: str
    service: str
    severity: str
    message: str = Field(min_length=1, max_length=4000)


class IncidentRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=100)
    events: list[LogEvent] = Field(min_length=1, max_length=500)


def get_models():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    rows = load_dataset(DATASET_PATH)
    models = train_models(rows)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    return models


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/incidents")
def incidents() -> list[dict[str, str | int]]:
    return [{key: row[key] for key in ("incident_id", "is_anomaly", "root_cause")} for row in load_dataset(DATASET_PATH)]


@app.get("/incidents/{incident_id}/analyze")
def analyze_incident(incident_id: str) -> dict:
    incident = next((row for row in load_dataset(DATASET_PATH) if row["incident_id"] == incident_id), None)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return analyze(incident, get_models())


@app.post("/analyze")
def analyze_payload(payload: IncidentRequest) -> dict:
    """Analyze caller-supplied logs. This endpoint never performs remediation."""
    incident = {"incident_id": payload.incident_id, "events": [event.model_dump() for event in payload.events]}
    return analyze(incident, get_models())
