from .models import ValidationMetrics

def compute_metrics(comparisons):
    comparisons = tuple(comparisons)
    total = len(comparisons)
    automatic = [c for c in comparisons if c.prediction in ("IN", "OUT")]
    correct = sum(1 for c in automatic if c.correct)
    incorrect = len(automatic) - correct
    reviews = sum(1 for c in comparisons if c.category == "REVIEW")
    misses = sum(1 for c in comparisons if c.category == "MISSING")
    false_in = sum(1 for c in comparisons if c.category == "FALSE_IN")
    false_out = sum(1 for c in comparisons if c.category == "FALSE_OUT")
    avg_conf = sum(c.confidence for c in automatic) / len(automatic) if automatic else 0.0
    avg_latency = sum(c.latency_ms for c in automatic) / len(automatic) if automatic else 0.0

    return ValidationMetrics(
        total_truth_events=total,
        automatic_calls=len(automatic),
        correct_automatic=correct,
        incorrect_automatic=incorrect,
        reviews=reviews,
        misses=misses,
        false_in=false_in,
        false_out=false_out,
        automatic_accuracy=(correct / len(automatic)) if automatic else 0.0,
        coverage=(len(automatic) / total) if total else 0.0,
        review_rate=(reviews / total) if total else 0.0,
        miss_rate=(misses / total) if total else 0.0,
        average_confidence=avg_conf,
        average_latency_ms=avg_latency,
    )
