import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def database_url() -> str:
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Start the service through Docker "
            "Compose or set a local PostgreSQL connection URL."
        )

    return url


def initialize_database() -> None:
    """Create the incident-decision table if it does not already exist."""
    with psycopg.connect(database_url(), autocommit=True) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_decisions (
                id BIGSERIAL PRIMARY KEY,
                incident_id TEXT NOT NULL,
                source TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                top_similarity DOUBLE PRECISION NOT NULL,
                evidence_labels JSONB NOT NULL,
                policy JSONB NOT NULL,
                evidence JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_decisions_created_at
            ON incident_decisions (created_at DESC)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_incident_decisions_incident_id
            ON incident_decisions (incident_id)
            """
        )


def save_incident_decision(
    *,
    incident_id: str,
    source: str,
    recommendation: str,
    top_similarity: float,
    evidence_labels: list[int],
    policy: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> int:
    """
    Persist a completed triage decision and return its database record ID.
    """
    with psycopg.connect(database_url(), autocommit=True) as connection:
        result = connection.execute(
            """
            INSERT INTO incident_decisions (
                incident_id,
                source,
                recommendation,
                top_similarity,
                evidence_labels,
                policy,
                evidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                incident_id,
                source,
                recommendation,
                top_similarity,
                Jsonb(evidence_labels),
                Jsonb(policy),
                Jsonb(evidence),
            ),
        )

        return int(result.fetchone()[0])