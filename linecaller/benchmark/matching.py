from __future__ import annotations

from statistics import median

from .models import EventMatch


def match_events(
    expected: list[dict],
    detected: list[dict],
    *,
    frame_tolerance: int = 2,
) -> tuple[tuple[EventMatch, ...], int, int]:

    used_detected: set[int] = set()
    matches: list[EventMatch] = []

    for exp in expected:
        ef = int(exp["frame"])
        best_index = None
        best_error = None

        for i, det in enumerate(detected):
            if i in used_detected:
                continue

            df = int(det["bounce_frame"])
            err = abs(df - ef)

            if err <= frame_tolerance:
                if best_error is None or err < best_error:
                    best_index = i
                    best_error = err

        if best_index is None:
            continue

        used_detected.add(best_index)
        det = detected[best_index]

        expected_decision = exp.get("decision")
        detected_decision = det.get("decision")

        decision_match = None
        if expected_decision is not None and detected_decision is not None:
            decision_match = str(expected_decision) == str(detected_decision)

        matches.append(
            EventMatch(
                expected_frame=ef,
                detected_frame=int(det["bounce_frame"]),
                frame_error=int(best_error),
                expected_decision=expected_decision,
                detected_decision=detected_decision,
                decision_match=decision_match,
            )
        )

    false_positives = len(detected) - len(used_detected)
    misses = len(expected) - len(matches)

    return tuple(matches), false_positives, misses
