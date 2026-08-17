import random


def select_balanced_support_set(
    incidents: list[dict],
    total_size: int,
    seed: int = 7,
) -> list[dict]:
    """
    Select a reproducible, class-balanced labelled support set.

    The returned incidents remain in their original chronological order
    after sampling. This is useful for few-shot adaptation experiments.
    """
    if total_size <= 0:
        raise ValueError("total_size must be positive.")

    if total_size % 2 != 0:
        raise ValueError("total_size must be even for balanced sampling.")

    normal_incidents = [
        incident
        for incident in incidents
        if int(incident["is_anomaly"]) == 0
    ]
    anomaly_incidents = [
        incident
        for incident in incidents
        if int(incident["is_anomaly"]) == 1
    ]

    per_class = total_size // 2

    if len(normal_incidents) < per_class:
        raise ValueError("Not enough normal incidents in support pool.")

    if len(anomaly_incidents) < per_class:
        raise ValueError("Not enough anomaly incidents in support pool.")

    random_generator = random.Random(seed)

    selected_ids = {
        id(incident)
        for incident in random_generator.sample(
            normal_incidents,
            per_class,
        )
        + random_generator.sample(
            anomaly_incidents,
            per_class,
        )
    }

    # Preserve original chronological order from the supplied incident list.
    return [
        incident
        for incident in incidents
        if id(incident) in selected_ids
    ]