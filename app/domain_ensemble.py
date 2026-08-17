from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class BlendSelection:
    source_weight: float
    anomaly_threshold: float
    validation_f1: float


def blend_probabilities(
    source_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    source_weight: float,
) -> np.ndarray:
    """
    Blend source and target anomaly probabilities.

    source_weight=0.0 means target-only.
    source_weight=1.0 means source-only.
    """
    if not 0.0 <= source_weight <= 1.0:
        raise ValueError("source_weight must be between 0 and 1")

    source = np.asarray(source_probabilities, dtype=float)
    target = np.asarray(target_probabilities, dtype=float)

    if source.shape != target.shape:
        raise ValueError("source_probabilities and target_probabilities must have the same shape")

    return source_weight * source + (1.0 - source_weight) * target


def select_blend_from_validation(
    labels: np.ndarray,
    source_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    source_weights: list[float],
    thresholds: list[float],
) -> BlendSelection:
    """
    Select a source contribution and anomaly threshold using validation F1 only.

    When two settings have identical F1, prefer less source influence.
    """
    labels = np.asarray(labels, dtype=int)

    best: BlendSelection | None = None

    for source_weight in source_weights:
        blended = blend_probabilities(
            source_probabilities,
            target_probabilities,
            source_weight,
        )

        for threshold in thresholds:
            predictions = (blended >= threshold).astype(int)
            score = f1_score(labels, predictions, zero_division=0)

            candidate = BlendSelection(
                source_weight=source_weight,
                anomaly_threshold=threshold,
                validation_f1=float(score),
            )

            if best is None:
                best = candidate
                continue

            if candidate.validation_f1 > best.validation_f1:
                best = candidate
            elif (
                candidate.validation_f1 == best.validation_f1
                and candidate.source_weight < best.source_weight
            ):
                best = candidate

    if best is None:
        raise ValueError("source_weights and thresholds must not be empty")

    return best


def probability_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Return decision confidence for anomaly probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    return np.maximum(probabilities, 1.0 - probabilities)