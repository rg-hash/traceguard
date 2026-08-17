import numpy as np


def fit_similarity_ood_threshold(
    source_validation_similarities: np.ndarray,
    allowed_source_ood_rate: float = 0.05,
) -> float:
    """
    Learn a source-domain similarity threshold without target-domain labels.

    Example:
    allowed_source_ood_rate=0.05 means that at most the lowest 5% of
    HDFS validation incidents are treated as out-of-distribution.
    """
    if not 0.0 < allowed_source_ood_rate < 1.0:
        raise ValueError(
            "allowed_source_ood_rate must be between 0 and 1."
        )

    if len(source_validation_similarities) == 0:
        raise ValueError(
            "source_validation_similarities cannot be empty."
        )

    return float(
        np.quantile(
            source_validation_similarities,
            allowed_source_ood_rate,
        )
    )


def is_in_distribution(
    maximum_source_similarity: np.ndarray,
    similarity_threshold: float,
) -> np.ndarray:
    """
    Return True when an incident resembles the source-domain evidence.

    A low similarity means the incident may come from an unfamiliar
    operational domain and should be routed to human review.
    """
    return maximum_source_similarity >= similarity_threshold