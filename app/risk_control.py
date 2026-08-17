from __future__ import annotations

from math import sqrt
from statistics import NormalDist


def wilson_upper_bound(
    errors: int,
    total: int,
    confidence: float = 0.95,
) -> float:
    """
    One-sided Wilson upper confidence bound for an error rate.

    Example:
    If 0 of 100 automated validation decisions are wrong, the
    true error rate is still not assumed to be exactly zero.
    """
    if total <= 0:
        return 1.0

    if not 0 <= errors <= total:
        raise ValueError("errors must be between 0 and total")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    observed_rate = errors / total
    z = NormalDist().inv_cdf(confidence)

    denominator = 1.0 + (z * z) / total
    center = (
        observed_rate + (z * z) / (2.0 * total)
    ) / denominator
    margin = (
        z
        * sqrt(
            (
                observed_rate * (1.0 - observed_rate)
                + (z * z) / (4.0 * total)
            )
            / total
        )
        / denominator
    )

    return min(1.0, center + margin)


def risk_is_acceptable(
    errors: int,
    total: int,
    maximum_risk: float,
    confidence: float = 0.95,
) -> bool:
    """Accept only if the conservative upper risk bound meets the target."""
    if not 0.0 <= maximum_risk <= 1.0:
        raise ValueError("maximum_risk must be between 0 and 1")

    return wilson_upper_bound(
        errors=errors,
        total=total,
        confidence=confidence,
    ) <= maximum_risk