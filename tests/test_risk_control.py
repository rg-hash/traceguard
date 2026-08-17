import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app.risk_control import risk_is_acceptable, wilson_upper_bound


def test_zero_errors_with_many_examples_has_low_upper_bound():
    assert wilson_upper_bound(errors=0, total=2000) < 0.05


def test_zero_errors_with_few_examples_is_still_uncertain():
    assert wilson_upper_bound(errors=0, total=10) > 0.05


def test_more_errors_increase_the_upper_bound():
    assert wilson_upper_bound(errors=5, total=100) > wilson_upper_bound(
        errors=0,
        total=100,
    )


def test_invalid_error_count_is_rejected():
    with pytest.raises(ValueError):
        wilson_upper_bound(errors=11, total=10)


def test_risk_acceptance_uses_conservative_bound():
    assert risk_is_acceptable(
        errors=0,
        total=2000,
        maximum_risk=0.05,
    )

    assert not risk_is_acceptable(
        errors=0,
        total=10,
        maximum_risk=0.05,
    )