import re


TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_$<>-]*")

STOP_WORDS = {
    "a", "an", "and", "at", "for", "from", "in", "of", "on",
    "the", "to", "with", "is", "was", "info",
}


def meaningful_tokens(text: str) -> set[str]:
    """Extract technical tokens useful for comparing log templates."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return {token for token in tokens if token not in STOP_WORDS}


def rank_evidence_lines(
    query_text: str,
    evidence_text: str,
    max_lines: int = 3,
) -> list[dict[str, float | str]]:
    """
    Select the most relevant normalized log templates from one retrieved
    incident, so the system can cite concrete evidence to the operator.
    """
    query_tokens = meaningful_tokens(query_text)
    candidates = []
    seen_templates: set[str] = set()
    for line in evidence_text.splitlines():
        line = line.strip()
        if line in seen_templates:
            continue

        seen_templates.add(line)
        if not line:
            continue

        line_tokens = meaningful_tokens(line)
        overlap = query_tokens & line_tokens

        if not overlap:
            continue

        jaccard = len(overlap) / len(query_tokens | line_tokens)
        query_coverage = len(overlap) / len(query_tokens)

        score = 0.6 * query_coverage + 0.4 * jaccard

        candidates.append(
            {
                "template": line,
                "score": round(score, 4),
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)

    return candidates[:max_lines]