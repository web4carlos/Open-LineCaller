from .models import ValidationComparison, ValidationPrediction

def compare_events(truth_events, predictions):
    pred_map = {p.event_id: p for p in predictions}
    results = []

    for truth in truth_events:
        p = pred_map.get(truth.event_id)
        if p is None:
            p = ValidationPrediction(
                event_id=truth.event_id,
                frame=truth.frame,
                prediction="MISSING",
            )

        if p.prediction in ("IN", "OUT"):
            correct = p.prediction == truth.truth
            if correct:
                category = "CORRECT"
            elif truth.truth == "OUT" and p.prediction == "IN":
                category = "FALSE_IN"
            else:
                category = "FALSE_OUT"
        elif p.prediction == "REVIEW":
            correct = False
            category = "REVIEW"
        else:
            correct = False
            category = "MISSING"

        results.append(
            ValidationComparison(
                event_id=truth.event_id,
                truth=truth.truth,
                prediction=p.prediction,
                correct=correct,
                category=category,
                confidence=float(p.confidence),
                latency_ms=float(p.latency_ms),
            )
        )

    return tuple(results)
