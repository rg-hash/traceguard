from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.agentic_explanation import (
    get_agentic_explanation_service,
)
from app.bgl_triage import get_bgl_triage_service
from app.config import DATASET_PATH, MODEL_PATH
from app.data import load_dataset
from app.database import (
    get_organization_profile,
    initialize_database,
    list_incident_decisions,
    list_investigation_feedback,
    list_organization_profiles,
    save_incident_decision,
    save_investigation_feedback,
    save_organization_profile,
)
from app.evidence import rank_evidence_lines
from app.hybrid_retrieval import HybridRetriever
from app.investigation import (
    get_investigation_service,
    get_organization_investigation_service,
)
from app.log_normalization import incident_to_text
from app.metrics import (
    BGL_DECISIONS_PERSISTED,
    BGL_HUMAN_REVIEWS,
    BGL_TRIAGE_FAILURES,
    BGL_TRIAGE_LATENCY_SECONDS,
    BGL_TRIAGE_REQUESTS,
    HDFS_DECISIONS_PERSISTED,
    HDFS_RETRIEVAL_FAILURES,
    HDFS_RETRIEVAL_LATENCY_SECONDS,
    HDFS_RETRIEVAL_REQUESTS,
)
from app.ml import analyze, train_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HDFS_TRAIN_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data/index/hdfs_hybrid_train_documents.jsonl"
)

HDFS_SEMANTIC_WEIGHT = 0.75
HDFS_MINIMUM_HYBRID_SCORE = 0.05
HDFS_TOP_K = 3


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare required services before the API accepts traffic."""
    initialize_database()
    get_hdfs_retriever()
    get_bgl_triage_service()
    get_investigation_service()
    yield


app = FastAPI(
    title="TraceGuard",
    version="1.0.0",
    description="Evidence-grounded AIOps incident analysis.",
    lifespan=lifespan,
)

app.mount(
    "/dashboard",
    StaticFiles(
        directory=PROJECT_ROOT / "static",
        html=True,
    ),
    name="dashboard",
)


class LogEvent(BaseModel):
    timestamp: str = ""
    service: str = ""
    severity: str = ""
    message: str = Field(min_length=1, max_length=4000)


class IncidentRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=100)
    events: list[LogEvent] = Field(min_length=1, max_length=500)


class TriageEvidence(BaseModel):
    historical_incident_id: str
    historical_label: str
    similarity: float | None = None
    cited_templates: list[dict[str, Any]] = Field(
        default_factory=list
    )


class TriageContext(BaseModel):
    source: str = "bgl_evidence_grounded_triage"

    classifier_anomaly_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    classifier_prediction: str | None = None

    decision_checks: dict[str, Any] = Field(
        default_factory=dict
    )

    evidence: list[TriageEvidence] = Field(
        default_factory=list
    )


class InvestigationRequest(IncidentRequest):
    organization_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern="^[a-z0-9][a-z0-9-]*$",
    )

    triage_recommendation: str = Field(
        default="NEEDS_HUMAN_REVIEW",
        pattern="^(LIKELY_ANOMALY|NEEDS_HUMAN_REVIEW)$",
    )

    triage_context: TriageContext | None = None


class ServiceDefinition(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100,
        pattern="^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
    )
    description: str = Field(default="", max_length=2000)
    owner: str = Field(default="", max_length=200)
    dependencies: list[str] = Field(default_factory=list, max_length=50)


class KnowledgeDocument(BaseModel):
    id: str = Field(min_length=2, max_length=100)
    kind: str = Field(
        pattern="^(incident|runbook)$"
    )
    title: str = Field(min_length=3, max_length=500)
    service: str = Field(default="shared", max_length=100)
    symptoms: list[str] = Field(default_factory=list, max_length=50)
    root_cause: str | None = Field(default=None, max_length=200)
    resolution: str = Field(default="", max_length=5000)
    steps: list[str] = Field(default_factory=list, max_length=30)
    tags: list[str] = Field(default_factory=list, max_length=50)


class DeploymentRecord(BaseModel):
    id: str = Field(min_length=2, max_length=100)
    service: str = Field(min_length=2, max_length=100)
    commit: str = Field(default="", max_length=100)
    timestamp: str = Field(default="", max_length=100)
    summary: str = Field(default="", max_length=2000)


class OrganizationOnboardingRequest(BaseModel):
    organization_id: str = Field(
        min_length=2,
        max_length=100,
        pattern="^[a-z0-9][a-z0-9-]*$",
    )
    display_name: str = Field(min_length=2, max_length=200)
    description: str = Field(default="", max_length=2000)
    services: list[ServiceDefinition] = Field(
        min_length=1,
        max_length=100,
    )
    knowledge: list[KnowledgeDocument] = Field(
        default_factory=list,
        max_length=1000,
    )
    deployments: list[DeploymentRecord] = Field(
        default_factory=list,
        max_length=1000,
    )

class InvestigationFeedbackRequest(BaseModel):
    incident_id: str = Field(
        min_length=1,
        max_length=100,
    )

    organization_id: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern="^[a-z0-9][a-z0-9-]*$",
    )

    triage_recommendation: str = Field(
        pattern="^(LIKELY_ANOMALY|NEEDS_HUMAN_REVIEW)$",
    )

    final_anomaly_label: str | None = Field(
        default=None,
        pattern="^(ANOMALY|NORMAL|UNCERTAIN)$",
    )

    incident_events: list[LogEvent] = Field(
        min_length=1,
        max_length=500,
    )

    hypothesis: str | None = Field(
        default=None,
        max_length=200,
    )

    hypothesis_accepted: bool | None = None

    confirmed_root_cause: str | None = Field(
        default=None,
        max_length=200,
    )

    resolution: str | None = Field(
        default=None,
        max_length=5000,
    )

    usefulness_rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    reviewer_note: str | None = Field(
        default=None,
        max_length=5000,
    )

@lru_cache
def get_hdfs_retriever() -> HybridRetriever:
    """
    Load historical HDFS training evidence once per API process.

    Only training incidents are indexed. Validation and test data
    are never included in the deployed evidence index.
    """
    if not HDFS_TRAIN_DOCUMENTS_PATH.exists():
        raise RuntimeError(
            "HDFS retrieval evidence is missing. Run "
            "scripts/build_hdfs_hybrid_index.py first."
        )

    with HDFS_TRAIN_DOCUMENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
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

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    """Return recently persisted triage decisions."""
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
    return [
        {
            key: row[key]
            for key in (
                "incident_id",
                "is_anomaly",
                "root_cause",
            )
        }
        for row in load_dataset(DATASET_PATH)
    ]


@app.get("/incidents/{incident_id}/analyze")
def analyze_incident(incident_id: str) -> dict:
    incident = next(
        (
            row
            for row in load_dataset(DATASET_PATH)
            if row["incident_id"] == incident_id
        ),
        None,
    )

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return analyze(incident, get_models())


@app.post("/analyze")
def analyze_payload(payload: IncidentRequest) -> dict:
    """Analyze caller-supplied logs. Never performs remediation."""
    incident = {
        "incident_id": payload.incident_id,
        "events": [
            event.model_dump()
            for event in payload.events
        ],
    }

    return analyze(incident, get_models())


@app.post("/investigate")
def investigate(payload: InvestigationRequest) -> dict:
    """
    Create a read-only, evidence-grounded debugging plan.

    The endpoint preserves the existing BGL/HDFS triage evidence.
    It never runs commands, restarts services, or changes production.
    """
    if payload.organization_id:
        profile = get_organization_profile(payload.organization_id)

        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Organization onboarding profile was not found.",
            )

        investigation_service = (
            get_organization_investigation_service(
                payload.organization_id,
                int(profile["knowledge_version"]),
            )
        )
    else:
        investigation_service = get_investigation_service()

    result = investigation_service.investigate(
        incident_id=payload.incident_id,
        events=[
            event.model_dump()
            for event in payload.events
        ],
        triage_recommendation=payload.triage_recommendation,
        triage_context=(
            payload.triage_context.model_dump()
            if payload.triage_context
            else {}
        ),
    )

    result["organization_id"] = payload.organization_id

    return result


@app.post("/investigate/agentic")
def investigate_agentically(
    payload: InvestigationRequest,
) -> dict:
    """
    Add an optional LLM explanation to a safe investigation result.

    The LLM has no operational tools. Deterministic validation rejects any
    citation outside TraceGuard's already retrieved evidence ledger.
    """
    result = investigate(payload)
    result["agentic_explanation"] = (
        get_agentic_explanation_service().explain(result)
    )

    return result


@app.post("/organizations/onboard")
def onboard_organization(
    payload: OrganizationOnboardingRequest,
) -> dict[str, int | str]:
    """
    Store approved product architecture and operational knowledge.

    The data becomes an isolated RAG corpus only for this organization. This
    endpoint does not crawl private systems or automatically train a model.
    """
    try:
        knowledge_version = save_organization_profile(
            organization_id=payload.organization_id,
            display_name=payload.display_name,
            description=payload.description,
            services=[
                service.model_dump()
                for service in payload.services
            ],
            knowledge=[
                document.model_dump()
                for document in payload.knowledge
            ],
            deployments=[
                deployment.model_dump()
                for deployment in payload.deployments
            ],
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to store organization onboarding data.",
        ) from error

    return {
        "organization_id": payload.organization_id,
        "knowledge_version": knowledge_version,
        "status": "onboarded",
    }


@app.get("/organizations")
def organizations(limit: int = 50) -> list[dict]:
    """List onboarded organizations without exposing their private corpus."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100.",
        )

    try:
        return list_organization_profiles(limit=limit)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to load organization profiles.",
        ) from error


@app.post("/investigations/feedback")
def submit_investigation_feedback(
    payload: InvestigationFeedbackRequest,
) -> dict[str, int | str]:
    """
    Store verified human feedback after an investigation.

    Feedback becomes approved data for later evaluation and controlled
    retraining. It never triggers automatic model retraining.
    """
    try:
        feedback_id = save_investigation_feedback(
            incident_id=payload.incident_id,
            organization_id=payload.organization_id,
            triage_recommendation=(
                payload.triage_recommendation
            ),
            final_anomaly_label=(
                payload.final_anomaly_label
            ),
            incident_events=[
                event.model_dump()
                for event in payload.incident_events
            ],
            hypothesis=payload.hypothesis,
            hypothesis_accepted=(
                payload.hypothesis_accepted
            ),
            confirmed_root_cause=(
                payload.confirmed_root_cause
            ),
            resolution=payload.resolution,
            usefulness_rating=payload.usefulness_rating,
            reviewer_note=payload.reviewer_note,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to store investigation feedback.",
        ) from error

    return {
        "feedback_id": feedback_id,
        "status": "stored",
    }

@app.get("/investigations/feedback")
def get_investigation_feedback(
    incident_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return recent human feedback for the dashboard or evaluation."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100.",
        )

    try:
        return list_investigation_feedback(
            incident_id=incident_id,
            limit=limit,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to load investigation feedback.",
        ) from error

@app.post("/retrieve/hdfs")
def retrieve_hdfs_evidence(payload: IncidentRequest) -> dict:
    """
    Retrieve historical HDFS evidence for caller-supplied logs.

    This endpoint recommends triage only. It never remediates systems.
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

    evidence_labels = [
        match.label
        for match in matches
    ]

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
                    "Anomaly"
                    if match.label == 1
                    else "Normal"
                ),
                "hybrid_similarity": round(
                    match.score,
                    4,
                ),
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
        "top_hybrid_similarity": round(
            matches[0].score,
            4,
        ),
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
        HDFS_RETRIEVAL_FAILURES.labels(
            reason="database"
        ).inc()

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

    Automated recommendations require classifier confidence,
    strong evidence similarity, unanimous top-3 evidence,
    and agreement between classifier and evidence.
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
        BGL_TRIAGE_FAILURES.labels(
            reason="invalid_input"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        BGL_TRIAGE_FAILURES.labels(
            reason="analysis"
        ).inc()

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
        BGL_TRIAGE_FAILURES.labels(
            reason="database"
        ).inc()

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
