"""
Read-only, evidence-grounded incident investigation workflow.

The agent never executes commands, changes deployments, restarts
services, or remediates infrastructure. It returns a human-reviewed
debugging plan only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "incident_knowledge.json"
)

DEPLOYMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "deployments.json"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Candidate configuration. We will select/finalize it using only
# investigation_development.json, never the frozen holdout set.
SEMANTIC_WEIGHT = 0.70
LEXICAL_WEIGHT = 1.0 - SEMANTIC_WEIGHT

# An unsupported/weakly related incident must not become a hypothesis.
MINIMUM_HYPOTHESIS_EVIDENCE = 0.26

ROOT_CAUSES = {
    "database_connection_pool_exhaustion": [
        "database",
        "connection",
        "pool",
        "timeout",
        "refused",
    ],
    "dns_or_network_failure": [
        "dns",
        "host",
        "network",
        "upstream",
        "reset",
    ],
    "recent_deployment_regression": [
        "deployment",
        "release",
        "exception",
        "validation",
        "rollback",
    ],
}


def knowledge_to_text(item: dict[str, Any]) -> str:
    """Convert approved incident/runbook data into searchable text."""
    fields = [
        item.get("title", ""),
        item.get("service", ""),
        item.get("resolution", ""),
    ]

    fields.extend(item.get("symptoms", []))
    fields.extend(item.get("tags", []))

    return " ".join(
        str(value)
        for value in fields
        if value
    ).lower()


def events_to_text(events: list[dict[str, Any]]) -> str:
    """Convert active incident logs into one retrieval query."""
    return " ".join(
        str(event.get("message", ""))
        for event in events
    ).lower()


class InvestigationState(TypedDict, total=False):
    incident_id: str
    events: list[dict[str, Any]]

    # Original HDFS/BGL prediction and safety evidence.
    triage_recommendation: str
    triage_context: dict[str, Any]

    summary: dict[str, Any]
    evidence: list[dict[str, Any]]
    deployment_context: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    recommended_checks: list[dict[str, str]]
    decision: str


class InvestigationService:
    def __init__(
        self,
        knowledge: list[dict[str, Any]],
        deployments: list[dict[str, Any]],
        semantic_weight: float = SEMANTIC_WEIGHT,
        minimum_hypothesis_evidence: float = (
            MINIMUM_HYPOTHESIS_EVIDENCE
        ),
    ) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if not 0.0 <= minimum_hypothesis_evidence <= 1.0:
            raise ValueError(
                "minimum_hypothesis_evidence must be between 0 and 1."
            )

        self.knowledge = knowledge
        self.deployments = deployments
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        self.minimum_hypothesis_evidence = (
            minimum_hypothesis_evidence
        )

        self.knowledge_texts = [
            knowledge_to_text(item)
            for item in knowledge
        ]

        self.lexical_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
        )

        self.lexical_matrix = (
            self.lexical_vectorizer.fit_transform(
                self.knowledge_texts
            )
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.semantic_matrix = self.embedding_model.encode(
            self.knowledge_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.graph = self.build_graph()

    def summarize_incident(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        """Create a deterministic incident summary from raw logs."""
        events = state["events"]

        messages = [
            str(event.get("message", ""))
            for event in events
        ]

        services = [
            str(event.get("service", ""))
            for event in events
            if event.get("service")
        ]

        error_messages = [
            message
            for message in messages
            if re.search(
                r"error|fatal|exception|timeout|failed|refused",
                message,
                re.IGNORECASE,
            )
        ]

        normalized_errors = [
            re.sub(
                r"\d+",
                "<num>",
                message.lower(),
            )
            for message in error_messages
        ]

        timestamps = [
            str(event.get("timestamp", ""))
            for event in events
            if event.get("timestamp")
        ]

        return {
            "summary": {
                "affected_service": (
                    Counter(services).most_common(1)[0][0]
                    if services
                    else "unknown"
                ),
                "event_count": len(events),
                "error_count": len(error_messages),
                "first_seen": (
                    min(timestamps)
                    if timestamps
                    else None
                ),
                "last_seen": (
                    max(timestamps)
                    if timestamps
                    else None
                ),
                "dominant_errors": [
                    message
                    for message, _ in Counter(
                        normalized_errors
                    ).most_common(3)
                ],
            }
        }

    def retrieve_evidence(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        """
        Retrieve approved incident and runbook evidence using hybrid
        semantic and lexical similarity.
        """
        query = events_to_text(state["events"])

        lexical_query = (
            self.lexical_vectorizer.transform([query])
        )

        lexical_scores = cosine_similarity(
            lexical_query,
            self.lexical_matrix,
        ).ravel()

        semantic_query = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Vectors are normalized, so dot product is cosine similarity.
        semantic_scores = self.semantic_matrix @ semantic_query

        hybrid_scores = (
            self.semantic_weight * semantic_scores
            + self.lexical_weight * lexical_scores
        )

        top_indices = hybrid_scores.argsort()[::-1][:4]

        evidence = []

        for index in top_indices:
            if hybrid_scores[index] <= 0:
                continue

            item = self.knowledge[index]

            evidence.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "title": item["title"],
                    "hybrid_similarity": round(
                        float(hybrid_scores[index]),
                        4,
                    ),
                    "semantic_similarity": round(
                        float(semantic_scores[index]),
                        4,
                    ),
                    "lexical_similarity": round(
                        float(lexical_scores[index]),
                        4,
                    ),
                    "root_cause": item.get("root_cause"),
                    "resolution": item.get("resolution"),
                    "steps": item.get("steps", []),
                }
            )

        return {"evidence": evidence}

    def correlate_deployments(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        """Find versioned deployments for the affected service."""
        service = state["summary"]["affected_service"]

        deployment_context = [
            deployment
            for deployment in self.deployments
            if deployment["service"] == service
        ]

        return {
            "deployment_context": deployment_context
        }

    def rank_hypotheses(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        """
        Rank root-cause categories using transparent signals.

        score =
            45% direct log-pattern match
          + 40% hybrid incident-evidence score
          + 15% deployment correlation
        """
        log_text = events_to_text(state["events"])

        hypotheses = []

        for root_cause, keywords in ROOT_CAUSES.items():
            log_score = (
                sum(
                    keyword in log_text
                    for keyword in keywords
                )
                / len(keywords)
            )

            matching_evidence = [
                item
                for item in state.get("evidence", [])
                if item.get("root_cause") == root_cause
            ]

            evidence_score = max(
                (
                    item["hybrid_similarity"]
                    for item in matching_evidence
                ),
                default=0.0,
            )

            deployment_score = (
                1.0
                if (
                    root_cause
                    == "recent_deployment_regression"
                    and state.get("deployment_context")
                )
                else 0.0
            )

            score = (
                0.45 * log_score
                + 0.40 * evidence_score
                + 0.15 * deployment_score
            )

            if (
                evidence_score
                >= self.minimum_hypothesis_evidence
            ):
                hypotheses.append(
                    {
                        "cause": root_cause,
                        "confidence": round(
                            min(score, 0.99),
                            4,
                        ),
                        "evidence_ids": [
                            item["id"]
                            for item in matching_evidence
                        ],
                        "why": (
                            "Score combines direct log-pattern "
                            "matches, hybrid historical evidence, "
                            "and deployment context."
                        ),
                    }
                )

        hypotheses.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {"hypotheses": hypotheses[:3]}

    def create_investigation_plan(
        self,
        state: InvestigationState,
    ) -> dict[str, Any]:
        """
        Select only approved runbook checks for investigation.

        The agent does not execute these steps.
        """
        recommended_checks = []
        seen_steps = set()

        hypothesis_causes = {
            hypothesis["cause"]
            for hypothesis in state.get("hypotheses", [])
        }

        for item in state.get("evidence", []):
            if item["kind"] != "runbook":
                continue

            is_relevant = any(
                keyword in item["title"].lower()
                for cause in hypothesis_causes
                for keyword in ROOT_CAUSES[cause]
            )

            if not is_relevant:
                continue

            for step in item["steps"]:
                if step not in seen_steps:
                    recommended_checks.append(
                        {
                            "step": step,
                            "source_id": item["id"],
                        }
                    )

                    seen_steps.add(step)

        return {
            "recommended_checks": recommended_checks[:6],
            "decision": "ENGINEER_REVIEW_REQUIRED",
        }

    def build_graph(self):
        """Create the controlled LangGraph workflow."""
        graph = StateGraph(InvestigationState)

        graph.add_node(
            "summarize_incident",
            self.summarize_incident,
        )

        graph.add_node(
            "retrieve_evidence",
            self.retrieve_evidence,
        )

        graph.add_node(
            "correlate_deployments",
            self.correlate_deployments,
        )

        graph.add_node(
            "rank_hypotheses",
            self.rank_hypotheses,
        )

        graph.add_node(
            "create_investigation_plan",
            self.create_investigation_plan,
        )

        graph.add_edge(
            START,
            "summarize_incident",
        )

        graph.add_edge(
            "summarize_incident",
            "retrieve_evidence",
        )

        graph.add_edge(
            "retrieve_evidence",
            "correlate_deployments",
        )

        graph.add_edge(
            "correlate_deployments",
            "rank_hypotheses",
        )

        graph.add_edge(
            "rank_hypotheses",
            "create_investigation_plan",
        )

        graph.add_edge(
            "create_investigation_plan",
            END,
        )

        return graph.compile()

    def investigate(
        self,
        incident_id: str,
        events: list[dict[str, Any]],
        triage_recommendation: str,
        triage_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a safe investigation after HDFS/BGL triage."""
        result = self.graph.invoke(
            {
                "incident_id": incident_id,
                "events": events,
                "triage_recommendation": (
                    triage_recommendation
                ),
                "triage_context": triage_context or {},
            }
        )

        return {
            "incident_id": incident_id,
            "triage_recommendation": (
                triage_recommendation
            ),

            "triage_context": result.get(
                "triage_context",
                {},
            ),

            "retrieval_policy": {
                "method": (
                    "hybrid semantic and lexical retrieval"
                ),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "semantic_weight": self.semantic_weight,
                "lexical_weight": self.lexical_weight,
                "minimum_hypothesis_evidence": (
                    self.minimum_hypothesis_evidence
                ),
            },

            "safety_notice": (
                "Read-only investigation plan. "
                "No infrastructure action is executed; "
                "engineer review is required."
            ),

            "summary": result["summary"],
            "evidence": result.get("evidence", []),

            "deployment_context": result.get(
                "deployment_context",
                [],
            ),

            "hypotheses": result.get(
                "hypotheses",
                [],
            ),

            "recommended_checks": result.get(
                "recommended_checks",
                [],
            ),

            "decision": result["decision"],
        }


@lru_cache
def get_investigation_service() -> InvestigationService:
    knowledge = json.loads(
        KNOWLEDGE_PATH.read_text()
    )

    deployments = json.loads(
        DEPLOYMENTS_PATH.read_text()
    )

    return InvestigationService(
        knowledge=knowledge,
        deployments=deployments,
    )
