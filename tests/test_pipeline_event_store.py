import json

from linecaller.decision.models import (
    Decision,
    DecisionContext,
    DecisionResult,
)
from linecaller.pipeline.event_store import DecisionEventStore


def test_event_store_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"

    context = DecisionContext(
        bounce_frame=10,
        image_x=100.0,
        image_y=200.0,
        court_x_m=1.0,
        court_y_m=2.0,
        nearest_line="left_sideline",
        signed_distance_m=0.1,
        local_m_per_px=0.01,
        calibration_error_px=1.0,
        bounce_error_px=2.0,
        total_uncertainty_m=0.02,
        bounce_confidence=0.95,
        calibration_valid=True,
        reasons=(),
    )

    result = DecisionResult(
        decision=Decision.IN,
        confidence=0.9,
        context=context,
        explanation=("test",),
    )

    store = DecisionEventStore(path)
    store.write(result)
    store.close()

    data = json.loads(path.read_text().strip())
    assert data["decision"] == "IN"
    assert data["bounce_frame"] == 10
