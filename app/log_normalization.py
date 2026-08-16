import re
from typing import Any


TIMESTAMP_PREFIX = re.compile(r"^\d{6}\s+\d{6}\s+\d+\s+")
BLOCK_ID = re.compile(r"\bblk_-?\d+\b")
IP_WITH_PORT = re.compile(
    r"/?\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b"
)
IP_ADDRESS = re.compile(r"/?\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_NUMBER = re.compile(r"\b\d{3,}\b")
WHITESPACE = re.compile(r"\s+")


def normalize_log_message(message: str) -> str:
    """
    Remove identifiers that vary between incidents while retaining the
    error template and system component information.

    Example:
    '... Receiving block blk_271... src: /10.1.2.3:50010'
    becomes:
    'info dfs.datanode$dataxceiver: receiving block <block_id>
     src: <ip_port>'
    """
    text = message.strip()

    text = TIMESTAMP_PREFIX.sub("", text)
    text = BLOCK_ID.sub("<block_id>", text)
    text = IP_WITH_PORT.sub("<ip_port>", text)
    text = IP_ADDRESS.sub("<ip>", text)
    text = LONG_NUMBER.sub("<number>", text)

    text = WHITESPACE.sub(" ", text).strip().lower()

    return text


def incident_to_text(incident: dict[str, Any]) -> str:
    """
    Convert one structured incident into normalized retrieval text.

    Labels, root-cause fields, and incident IDs are intentionally excluded
    so retrieval cannot learn from ground-truth metadata.
    """
    messages = []

    for event in incident.get("events", []):
        if isinstance(event, dict):
            message = event.get("message", "")
        else:
            message = str(event)

        if message:
            messages.append(normalize_log_message(str(message)))

    return "\n".join(messages)