from linecaller.validation.event_matcher import (
    match_predictions_to_truth,
)
from linecaller.validation.models import (
    ValidationPrediction,
    ValidationTruthEvent,
)


def test_matcher_uses_nearest_frame():
    truth = [
        ValidationTruthEvent(
            "truth-a",
            "x.mp4",
            100,
            "OUT",
        )
    ]

    predictions = [
        ValidationPrediction(
            "prediction-98",
            98,
            "OUT",
            .99,
            40,
        )
    ]

    matched = match_predictions_to_truth(
        truth,
        predictions,
        frame_tolerance=3,
    )

    assert len(matched) == 1
    assert matched[0].event_id == "truth-a"


def test_matcher_respects_tolerance():
    truth = [
        ValidationTruthEvent(
            "truth-a",
            "x.mp4",
            100,
            "OUT",
        )
    ]

    predictions = [
        ValidationPrediction(
            "prediction-90",
            90,
            "OUT",
        )
    ]

    matched = match_predictions_to_truth(
        truth,
        predictions,
        frame_tolerance=3,
    )

    assert matched == ()
