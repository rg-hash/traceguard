import os
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def database_url() -> str:
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Start PostgreSQL locally "
            "or set a local PostgreSQL connection URL."
        )

    return url


def initialize_database() -> None:
    """
    Create TraceGuard tables and indexes if they do not exist.

    Existing installations keep their current triage records.
    PostgreSQL safely creates only missing tables/indexes.
    """
    with psycopg.connect(
        database_url(),
        autocommit=True,
    ) as connection:
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
            CREATE INDEX IF NOT EXISTS
            idx_incident_decisions_created_at
            ON incident_decisions (created_at DESC)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_incident_decisions_incident_id
            ON incident_decisions (incident_id)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS investigation_feedback (
                id BIGSERIAL PRIMARY KEY,

                incident_id TEXT NOT NULL,

                triage_recommendation TEXT NOT NULL,

                hypothesis TEXT,

                hypothesis_accepted BOOLEAN,

                confirmed_root_cause TEXT,

                resolution TEXT,

                usefulness_rating SMALLINT CHECK (
                    usefulness_rating IS NULL
                    OR usefulness_rating BETWEEN 1 AND 5
                ),

                reviewer_note TEXT,

                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_investigation_feedback_incident_id
            ON investigation_feedback (incident_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_investigation_feedback_created_at
            ON investigation_feedback (created_at DESC)
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
    """Persist a completed HDFS/BGL triage decision."""
    with psycopg.connect(
        database_url(),
        autocommit=True,
    ) as connection:
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


def list_incident_decisions(
    *,
    recommendation: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent persisted HDFS/BGL triage decisions."""
    conditions = []
    parameters: list[Any] = []

    if recommendation is not None:
        conditions.append("recommendation = %s")
        parameters.append(recommendation)

    if source is not None:
        conditions.append("source = %s")
        parameters.append(source)

    where_clause = ""

    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    parameters.append(limit)

    query = f"""
        SELECT
            id,
            incident_id,
            source,
            recommendation,
            top_similarity,
            evidence_labels,
            policy,
            evidence,
            created_at
        FROM incident_decisions
        {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
    """

    with psycopg.connect(
        database_url(),
        row_factory=dict_row,
    ) as connection:
        result = connection.execute(query, parameters)

        return list(result.fetchall())


def save_investigation_feedback(
    *,
    incident_id: str,
    triage_recommendation: str,
    hypothesis: str | None,
    hypothesis_accepted: bool | None,
    confirmed_root_cause: str | None,
    resolution: str | None,
    usefulness_rating: int | None,
    reviewer_note: str | None,
) -> int:
    """
    Persist human feedback after an investigation.

    This is human-labelled ground truth. It can later be used for
    evaluation, retrieval, and carefully reviewed model improvement.
    """
    with psycopg.connect(
        database_url(),
        autocommit=True,
    ) as connection:
        result = connection.execute(
            """
            INSERT INTO investigation_feedback (
                incident_id,
                triage_recommendation,
                hypothesis,
                hypothesis_accepted,
                confirmed_root_cause,
                resolution,
                usefulness_rating,
                reviewer_note
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                incident_id,
                triage_recommendation,
                hypothesis,
                hypothesis_accepted,
                confirmed_root_cause,
                resolution,
                usefulness_rating,
                reviewer_note,
            ),
        )

        return int(result.fetchone()[0])


def list_investigation_feedback(
    *,
    incident_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent engineer feedback for dashboard and evaluation."""
    conditions = []
    parameters: list[Any] = []

    if incident_id is not None:
        conditions.append("incident_id = %s")
        parameters.append(incident_id)

    where_clause = ""

    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    parameters.append(limit)

    query = f"""
        SELECT
            id,
            incident_id,
            triage_recommendation,
            hypothesis,
            hypothesis_accepted,
            confirmed_root_cause,
            resolution,
            usefulness_rating,
            reviewer_note,
            created_at
        FROM investigation_feedback
        {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
    """

    with psycopg.connect(
        database_url(),
        row_factory=dict_row,
    ) as connection:
        result = connection.execute(query, parameters)

        return list(result.fetchall())