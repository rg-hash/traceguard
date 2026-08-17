from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalFold:
    name: str
    support_end: int
    validation_end: int
    evaluation_end: int


def expanding_window_folds(total_incidents: int) -> list[TemporalFold]:
    """
    Create three chronological folds.

    Each fold uses only earlier data for support/validation and evaluates
    on a later, unseen development period.
    """
    if total_incidents < 100:
        raise ValueError("Need at least 100 incidents for temporal backtesting")

    boundaries = [
        ("fold_1", 0.40, 0.60, 0.70),
        ("fold_2", 0.50, 0.70, 0.80),
        ("fold_3", 0.60, 0.80, 1.00),
    ]

    folds = []

    for name, support_fraction, validation_fraction, evaluation_fraction in boundaries:
        support_end = int(total_incidents * support_fraction)
        validation_end = int(total_incidents * validation_fraction)
        evaluation_end = int(total_incidents * evaluation_fraction)

        if not 0 < support_end < validation_end < evaluation_end:
            raise ValueError(f"Invalid temporal boundaries for {name}")

        folds.append(
            TemporalFold(
                name=name,
                support_end=support_end,
                validation_end=validation_end,
                evaluation_end=evaluation_end,
            )
        )

    return folds