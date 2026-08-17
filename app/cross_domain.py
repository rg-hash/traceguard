import re
from typing import Any


# HDFS identifiers
HDFS_TIMESTAMP = re.compile(r"^\d{6}\s+\d{6}\s+\d+\s+")
HDFS_BLOCK_ID = re.compile(r"\bblk_-?\d+\b")

# Network and numeric identifiers
IP_WITH_PORT = re.compile(r"/?\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b")
IP_ADDRESS = re.compile(r"/?\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_VALUE = re.compile(r"\b0x[0-9a-fA-F]+\b")
LONG_NUMBER = re.compile(r"\b\d{3,}\b")

# BGL identifiers
BGL_DATE = re.compile(r"\b\d{4}\.\d{2}\.\d{2}\b")
BGL_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}(?:\.\d+)?\b")
BGL_NODE = re.compile(r"\bR\d+-M\d+-(?:N\d+|NB|NE)-C:J\d+-U\d+\b")

WHITESPACE = re.compile(r"\s+")


def normalize_cross_domain_message(message: str) -> str:
    """
    Normalize both HDFS and BGL logs without using labels or incident IDs.

    The aim is to preserve error templates and components while removing
    identifiers that cannot transfer meaningfully across systems.
    """
    text = message.strip()

    text = HDFS_TIMESTAMP.sub("", text)
    text = HDFS_BLOCK_ID.sub("<block_id>", text)

    text = BGL_TIMESTAMP.sub("<timestamp>", text)
    text = BGL_DATE.sub("<date>", text)
    text = BGL_NODE.sub("<node>", text)

    text = IP_WITH_PORT.sub("<ip_port>", text)
    text = IP_ADDRESS.sub("<ip>", text)
    text = HEX_VALUE.sub("<hex>", text)
    text = LONG_NUMBER.sub("<number>", text)

    return WHITESPACE.sub(" ", text).strip().lower()


def incident_to_cross_domain_text(incident: dict[str, Any]) -> str:
    """
    Convert a structured HDFS or BGL incident into normalized text.

    Deliberately excludes:
    - incident_id
    - is_anomaly
    - root_cause
    """
    normalized_messages = []

    for event in incident.get("events", []):
        if isinstance(event, dict):
            message = str(event.get("message", ""))
        else:
            message = str(event)

        if message:
            normalized_messages.append(
                normalize_cross_domain_message(message)
            )

    return "\n".join(normalized_messages)
