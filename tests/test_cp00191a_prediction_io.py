from linecaller.validation.models import (
    ValidationPrediction,
)
from linecaller.validation.prediction_io import (
    save_predictions_jsonl,
)
from linecaller.validation.io import (
    load_predictions_jsonl,
)


def test_prediction_roundtrip(tmp_path):
    path = tmp_path / "pred.jsonl"

    predictions = [
        ValidationPrediction(
            event_id="x",
            frame=10,
            prediction="OUT",
            confidence=.99,
            latency_ms=42,
        )
    ]

    save_predictions_jsonl(
        predictions,
        path,
    )

    loaded = load_predictions_jsonl(
        path
    )

    assert loaded == predictions
