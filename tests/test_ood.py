import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ood import (
    fit_similarity_ood_threshold,
    is_in_distribution,
)


def test_similarity_threshold_uses_source_lower_tail():
    source_validation_similarities = np.array(
        [0.50, 0.60, 0.70, 0.80, 0.90]
    )

    threshold = fit_similarity_ood_threshold(
        source_validation_similarities,
        allowed_source_ood_rate=0.20,
    )

    assert threshold == 0.58


def test_low_similarity_is_flagged_as_out_of_distribution():
    maximum_similarity = np.array([0.20, 0.58, 0.80])

    accepted = is_in_distribution(
        maximum_similarity,
        similarity_threshold=0.58,
    )

    assert accepted.tolist() == [False, True, True]