from linecaller.live.models import (
    LiveDecision,
    LiveEvent,
)
from linecaller.live.pipeline_models import (
    LivePipelineResult,
)
from linecaller.live.realtime_adapter import (
    LiveAdapterOutput,
)


def test_review_bounce_is_valid_annotation_candidate_source():
    result = LivePipelineResult(
        frame_number=50,
        tracking_status="TRACKING",
        confidence=.9,
        ball_x=100.0,
        ball_y=200.0,
        bounce_detected=True,
        decision="REVIEW",
        decision_confidence=.8,
        metadata={
            "reason": "calibration_required"
        },
    )

    event = LiveEvent(
        frame_number=50,
        decision=LiveDecision.REVIEW,
        confidence=.8,
        event_timestamp=1.0,
        metadata=result.metadata,
    )

    output = LiveAdapterOutput(
        result=result,
        processing_ms=10.0,
        event=event,
    )

    assert output.event is not None
    assert output.result.bounce_detected
    assert output.event.decision == LiveDecision.REVIEW
