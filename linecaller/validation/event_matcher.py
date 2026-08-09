from __future__ import annotations

from dataclasses import replace


def match_predictions_to_truth(
    truth_events,
    predictions,
    *,
    frame_tolerance: int = 3,
):
    """
    Returns predictions with event_id reassigned to matching truth event_id.

    Matching priority:
    1. exact event_id
    2. nearest unmatched prediction within frame tolerance
    """

    predictions = list(predictions)

    exact = {
        p.event_id: p
        for p in predictions
    }

    matched_prediction_ids = set()
    result = []

    for truth in truth_events:
        if truth.event_id in exact:
            p = exact[truth.event_id]
            result.append(p)
            matched_prediction_ids.add(id(p))
            continue

        candidates = [
            p
            for p in predictions
            if id(p) not in matched_prediction_ids
            and abs(
                int(p.frame) - int(truth.frame)
            ) <= int(frame_tolerance)
        ]

        if not candidates:
            continue

        candidate = min(
            candidates,
            key=lambda p: abs(
                int(p.frame) - int(truth.frame)
            ),
        )

        matched_prediction_ids.add(id(candidate))

        result.append(
            replace(
                candidate,
                event_id=truth.event_id,
            )
        )

    return tuple(result)
