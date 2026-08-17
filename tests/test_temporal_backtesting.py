import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.temporal_backtesting import expanding_window_folds


def test_expanding_folds_are_strictly_temporal():
    folds = expanding_window_folds(10_000)

    assert len(folds) == 3

    for fold in folds:
        assert 0 < fold.support_end < fold.validation_end < fold.evaluation_end


def test_final_fold_uses_latest_development_period():
    final_fold = expanding_window_folds(10_000)[-1]

    assert final_fold.support_end == 6000
    assert final_fold.validation_end == 8000
    assert final_fold.evaluation_end == 10000