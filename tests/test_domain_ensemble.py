import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pytest

from app.domain_ensemble import (
    blend_probabilities,
    probability_confidence,
    select_blend_from_validation,
)


def test_blend_endpoints_match_source_and_target():
    source = np.array([0.9, 0.1])
    target = np.array([0.2, 0.8])

    assert np.allclose(blend_probabilities(source, target, 0.0), target)
    assert np.allclose(blend_probabilities(source, target, 1.0), source)


def test_blend_rejects_invalid_weight_or_shapes():
    with pytest.raises(ValueError):
        blend_probabilities(np.array([0.1]), np.array([0.2]), 1.1)

    with pytest.raises(ValueError):
        blend_probabilities(np.array([0.1]), np.array([0.2, 0.3]), 0.5)


def test_validation_prefers_target_when_target_is_correct():
    labels = np.array([0, 0, 1, 1])
    source = np.array([0.9, 0.9, 0.1, 0.1])  # Completely wrong.
    target = np.array([0.1, 0.2, 0.8, 0.9])  # Completely correct.

    selection = select_blend_from_validation(
        labels,
        source,
        target,
        source_weights=[0.0, 0.5, 1.0],
        thresholds=[0.5],
    )

    assert selection.source_weight == 0.0
    assert selection.validation_f1 == 1.0


def test_confidence_is_high_at_probability_extremes():
    confidence = probability_confidence(np.array([0.1, 0.5, 0.9]))

    assert np.allclose(confidence, np.array([0.9, 0.5, 0.9]))