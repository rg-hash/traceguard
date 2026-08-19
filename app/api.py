from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import time
from app.config import DATASET_PATH, MODEL_PATH
from app.data import load_dataset
from app.ml import analyze, train_models
from app.evidence import rank_evidence_lines
from app.hybrid_retrieval import HybridRetriever
from app.log_normalization import incident_to_text
from app.database import (
    initialize_database,
    list_incident_decisions,
    save_incident_decision,
)
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from contextlib import asynccontextmanager
from app.bgl_triage import get_bgl_triage_service
from app.metrics import (
    HDFS_DECISIONS_PERSISTED,
    HDFS_RETRIEVAL_FAILURES,
    HDFS_RETRIEVAL_LATENCY_SECONDS,
    HDFS_RETRIEVAL_REQUESTS,
    BGL_DECISIONS_PERSISTED,
    BGL_HUMAN_REVIEWS,
    BGL_TRIAGE_FAILURES,
    BGL_TRIAGE_LATENCY_SECONDS,
    BGL_TRIAGE_REQUESTS,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare required services before the API accepts traffic."""
    initialize_database()
    get_hdfs_retriever()
    get_bgl_triage_service()
    yield


app = FastAPI(
    title="TraceGuard",
    version="1.0.0",
    description="Evidence-grounded AIOps incident analysis.",
    lifespan=lifespan,
)
class LogEvent(BaseModel):
    timestamp: str = ""
    service: str = ""
    severity: str = ""
    message: str = Field(min_length=1, max_length=4000)



class IncidentRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=100)
    events: list[LogEvent] = Field(min_length=1, max_length=500)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app.mount(
    "/dashboard",
    StaticFiles(
        directory=PROJECT_ROOT / "static",
        html=True,
    ),
    name="dashboard",
)

HDFS_TRAIN_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data/index/hdfs_hybrid_train_documents.jsonl"
)

HDFS_SEMANTIC_WEIGHT = 0.75
HDFS_MINIMUM_HYBRID_SCORE = 0.05
HDFS_TOP_K = 3


@lru_cache
def get_hdfs_retriever() -> HybridRetriever:
    """
    Load the historical HDFS training evidence once per API process.

    The index contains only the 1,200 historic training incidents used in
    the validated experiment. It never includes validation or test data.
    """
    if not HDFS_TRAIN_DOCUMENTS_PATH.exists():
        raise RuntimeError(
            "HDFS retrieval evidence is missing. Run "
            "scripts/build_hdfs_hybrid_index.py first."
        )

    with HDFS_TRAIN_DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        documents = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    retriever = HybridRetriever(
        semantic_weight=HDFS_SEMANTIC_WEIGHT
    )
    retriever.build_index(documents)

    return retriever

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

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus-compatible application metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/decisions")
def decisions(
    recommendation: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return recently persisted triage decisions for the operator dashboard.
    """
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100.",
        )

    try:
        return list_incident_decisions(
            recommendation=recommendation,
            source=source,
            limit=limit,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to load persisted decisions.",
        ) from error

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

@app.post("/retrieve/hdfs")
def retrieve_hdfs_evidence(payload: IncidentRequest) -> dict:
    """
    Retrieve historical HDFS evidence for a caller-supplied incident.

    This endpoint recommends triage only. It never remediates a system.
    """
    started_at = time.perf_counter()
    incident = {
        "incident_id": payload.incident_id,
        "events": [
            event.model_dump()
            for event in payload.events
        ],
    }

    query_text = incident_to_text(incident)

    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="No usable log messages were provided.",
        )

    retriever = get_hdfs_retriever()
    matches = retriever.search(
        query_text=query_text,
        top_k=HDFS_TOP_K,
    )

    evidence_labels = [match.label for match in matches]
    unanimous_evidence = len(set(evidence_labels)) == 1
    strong_similarity = (
        matches[0].score >= HDFS_MINIMUM_HYBRID_SCORE
    )

    if strong_similarity and unanimous_evidence:
        recommendation = (
            "LIKELY_ANOMALY"
            if evidence_labels[0] == 1
            else "LIKELY_NORMAL"
        )
    else:
        recommendation = "NEEDS_HUMAN_REVIEW"

    evidence = []

    for rank, match in enumerate(matches, start=1):
        evidence.append(
            {
                "rank": rank,
                "historical_incident_id": match.incident_id,
                "historical_label": (
                    "Anomaly" if match.label == 1 else "Normal"
                ),
                "hybrid_similarity": round(match.score, 4),
                "cited_templates": rank_evidence_lines(
                    query_text=query_text,
                    evidence_text=match.text,
                    max_lines=2,
                ),
            }
        )

        policy = {
        "semantic_weight": HDFS_SEMANTIC_WEIGHT,
        "lexical_weight": 1.0 - HDFS_SEMANTIC_WEIGHT,
        "minimum_hybrid_score": HDFS_MINIMUM_HYBRID_SCORE,
        "requires_unanimous_top_3_evidence": True,
    }

    response = {
        "incident_id": payload.incident_id,
        "recommendation": recommendation,
        "policy": policy,
        "top_hybrid_similarity": round(matches[0].score, 4),
        "evidence_labels": evidence_labels,
        "evidence": evidence,
    }

    try:
        decision_record_id = save_incident_decision(
            incident_id=payload.incident_id,
            source="hdfs_hybrid_retrieval",
            recommendation=recommendation,
            top_similarity=float(matches[0].score),
            evidence_labels=evidence_labels,
            policy=policy,
            evidence=evidence,
        )
    except Exception as error:
        HDFS_RETRIEVAL_FAILURES.labels(reason="database").inc()

        raise HTTPException(
            status_code=503,
            detail="Unable to persist the triage decision.",
        ) from error

    HDFS_RETRIEVAL_REQUESTS.labels(
        recommendation=recommendation
    ).inc()

    HDFS_RETRIEVAL_LATENCY_SECONDS.observe(
        time.perf_counter() - started_at
    )

    HDFS_DECISIONS_PERSISTED.inc()

    response["decision_record_id"] = decision_record_id

    return response

@app.post("/triage/bgl")
def triage_bgl(payload: IncidentRequest) -> dict:
    """
    Safely triage caller-provided BGL log events.

    The endpoint returns an automated recommendation only when the
    classifier and three retrieved historical evidence incidents pass
    the validated safety policy. It never performs remediation.
    """
    started_at = time.perf_counter()

    incident_events = [
        event.model_dump()
        for event in payload.events
    ]

    try:
        result = get_bgl_triage_service().triage(
            incident_id=payload.incident_id,
            events=incident_events,
        )
    except ValueError as error:
        BGL_TRIAGE_FAILURES.labels(reason="invalid_input").inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        BGL_TRIAGE_FAILURES.labels(reason="analysis").inc()

        raise HTTPException(
            status_code=503,
            detail="Unable to analyze the BGL incident.",
        ) from error

    try:
        decision_record_id = save_incident_decision(
            incident_id=payload.incident_id,
            source="bgl_evidence_grounded_triage",
            recommendation=result["recommendation"],
            top_similarity=float(
                result["decision_checks"]["top_similarity"]
            ),
            evidence_labels=result["evidence_labels"],
            policy=result["policy"],
            evidence=result["evidence"],
        )
    except Exception as error:
        BGL_TRIAGE_FAILURES.labels(reason="database").inc()

        raise HTTPException(
            status_code=503,
            detail="Unable to persist the BGL triage decision.",
        ) from error

    BGL_TRIAGE_REQUESTS.labels(
        recommendation=result["recommendation"]
    ).inc()

    BGL_TRIAGE_LATENCY_SECONDS.observe(
        time.perf_counter() - started_at
    )

    BGL_DECISIONS_PERSISTED.inc()

    if result["recommendation"] == "NEEDS_HUMAN_REVIEW":
        BGL_HUMAN_REVIEWS.inc()

    result["decision_record_id"] = decision_record_id

    return result