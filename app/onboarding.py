"""Helpers for converting an organization's approved data into RAG evidence."""

from __future__ import annotations

from typing import Any


def _unique_strings(values: list[Any]) -> list[str]:
    """Return non-empty strings once, preserving the submitted order."""
    seen: set[str] = set()
    result = []

    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def build_organization_knowledge(
    services: list[dict[str, Any]],
    knowledge: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a tenant-specific retrieval corpus.

    Service architecture is deliberately converted into searchable evidence,
    so an investigation can cite dependencies even when no past incident is
    an exact match. Only caller-approved onboarding data is included.
    """
    architecture_documents = []

    for service in services:
        name = str(service["name"]).strip()
        dependencies = _unique_strings(
            service.get("dependencies", [])
        )

        description = str(service.get("description", "")).strip()
        owner = str(service.get("owner", "")).strip()

        architecture_documents.append(
            {
                "id": f"ARCH-{name}",
                "kind": "architecture",
                "title": f"Service context for {name}",
                "service": name,
                "symptoms": [],
                "resolution": description,
                "tags": _unique_strings(
                    [name, owner, *dependencies]
                ),
                "dependencies": dependencies,
            }
        )

    approved_documents = []

    for document in knowledge:
        # Copy only the fields used by the investigation workflow. This keeps
        # the retrieval corpus explicit and prevents arbitrary payload data
        # from silently influencing hypotheses.
        approved_documents.append(
            {
                "id": str(document["id"]).strip(),
                "kind": str(document["kind"]).strip(),
                "title": str(document["title"]).strip(),
                "service": str(document.get("service", "shared")).strip(),
                "symptoms": _unique_strings(
                    document.get("symptoms", [])
                ),
                "root_cause": document.get("root_cause"),
                "resolution": str(document.get("resolution", "")).strip(),
                "steps": _unique_strings(document.get("steps", [])),
                "tags": _unique_strings(document.get("tags", [])),
            }
        )

    return [*architecture_documents, *approved_documents]
