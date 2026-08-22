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
    PostgreSQL safely creates only missing tables, columns, and indexes.
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

                organization_id TEXT,

                triage_recommendation TEXT NOT NULL,

                final_anomaly_label TEXT CHECK (
                    final_anomaly_label IS NULL
                    OR final_anomaly_label IN (
                        'ANOMALY',
                        'NORMAL',
                        'UNCERTAIN'
                    )
                ),
                incident_events JSONB NOT NULL DEFAULT '[]'::jsonb,

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

        # Supports databases created before organization-specific
        # feedback and verified anomaly labels were added.
        connection.execute(
            """
            ALTER TABLE investigation_feedback
            ADD COLUMN IF NOT EXISTS organization_id TEXT
            """
        )

        connection.execute(
            """
            ALTER TABLE investigation_feedback
            ADD COLUMN IF NOT EXISTS incident_events JSONB
            NOT NULL DEFAULT '[]'::jsonb
            """
        )

        connection.execute(
            """
            ALTER TABLE investigation_feedback
            ADD COLUMN IF NOT EXISTS final_anomaly_label TEXT
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
            idx_investigation_feedback_organization_id
            ON investigation_feedback (organization_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_investigation_feedback_final_anomaly_label
            ON investigation_feedback (final_anomaly_label)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_investigation_feedback_created_at
            ON investigation_feedback (created_at DESC)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS organizations (
                organization_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                services JSONB NOT NULL,
                knowledge JSONB NOT NULL,
                deployments JSONB NOT NULL,
                knowledge_version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_organizations_updated_at
            ON organizations (updated_at DESC)
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
    organization_id: str | None,
    triage_recommendation: str,
    final_anomaly_label: str | None,
    incident_events: list[dict[str, Any]],
    hypothesis: str | None,
    hypothesis_accepted: bool | None,
    confirmed_root_cause: str | None,
    resolution: str | None,
    usefulness_rating: int | None,
    reviewer_note: str | None,
) -> int:
    """
    Persist verified human feedback after an investigation.

    This data is not used for immediate online retraining. It becomes
    a reviewed dataset for later evaluation and controlled retraining.
    """
    with psycopg.connect(
        database_url(),
        autocommit=True,
    ) as connection:
        result = connection.execute(
            """
            INSERT INTO investigation_feedback (
                incident_id,
                organization_id,
                triage_recommendation,
                final_anomaly_label,
                incident_events,
                hypothesis,
                hypothesis_accepted,
                confirmed_root_cause,
                resolution,
                usefulness_rating,
                reviewer_note
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                incident_id,
                organization_id,
                triage_recommendation,
                final_anomaly_label,
                Jsonb(incident_events),
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
            organization_id,
            triage_recommendation,
            final_anomaly_label,
            incident_events,
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


def list_verified_learning_feedback(
    *,
    organization_id: str,
    limit: int = 10_000,
) -> list[dict[str, Any]]:
    """
    Return only human-confirmed records that are eligible for offline learning.

    Uncertain records remain available for investigation review but are never
    used as training labels. This function is intentionally used by an offline
    script, not by the live prediction path.
    """
    with psycopg.connect(
        database_url(),
        row_factory=dict_row,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                incident_id,
                organization_id,
                incident_events,
                final_anomaly_label,
                confirmed_root_cause,
                created_at
            FROM investigation_feedback
            WHERE organization_id = %s
              AND final_anomaly_label IN ('ANOMALY', 'NORMAL')
              AND jsonb_array_length(incident_events) > 0
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (organization_id, limit),
        )

        return list(result.fetchall())


def save_organization_profile(
    *,
    organization_id: str,
    display_name: str,
    description: str,
    services: list[dict[str, Any]],
    knowledge: list[dict[str, Any]],
    deployments: list[dict[str, Any]],
) -> int:
    """Create or replace one organization's approved investigation context."""
    with psycopg.connect(
        database_url(),
        autocommit=True,
    ) as connection:
        result = connection.execute(
            """
            INSERT INTO organizations (
                organization_id,
                display_name,
                description,
                services,
                knowledge,
                deployments
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (organization_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                services = EXCLUDED.services,
                knowledge = EXCLUDED.knowledge,
                deployments = EXCLUDED.deployments,
                knowledge_version = organizations.knowledge_version + 1,
                updated_at = NOW()
            RETURNING knowledge_version
            """,
            (
                organization_id,
                display_name,
                description,
                Jsonb(services),
                Jsonb(knowledge),
                Jsonb(deployments),
            ),
        )

        return int(result.fetchone()[0])


def get_organization_profile(
    organization_id: str,
) -> dict[str, Any] | None:
    """Return the latest approved onboarding profile for one organization."""
    with psycopg.connect(
        database_url(),
        row_factory=dict_row,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                organization_id,
                display_name,
                description,
                services,
                knowledge,
                deployments,
                knowledge_version,
                created_at,
                updated_at
            FROM organizations
            WHERE organization_id = %s
            """,
            (organization_id,),
        )

        return result.fetchone()


def list_organization_profiles(
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List safe onboarding metadata without returning knowledge contents."""
    with psycopg.connect(
        database_url(),
        row_factory=dict_row,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                organization_id,
                display_name,
                description,
                knowledge_version,
                jsonb_array_length(services) AS service_count,
                jsonb_array_length(knowledge)
                    AS knowledge_document_count,
                jsonb_array_length(deployments) AS deployment_count,
                created_at,
                updated_at
            FROM organizations
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )

        return list(result.fetchall())
