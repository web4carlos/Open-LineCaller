import numpy as np

from linecaller.live.models import LiveDecision
from linecaller.live.perception_adapter import LivePerceptionAdapter
from linecaller.live.pipeline_models import LivePipelineResult
from linecaller.live.realtime_adapter import RealTimeOfficiatingAdapter


class FakePerception(LivePerceptionAdapter):
    def __init__(self, result):
        self.result = result

    def process(self, *, frame_number, frame, timestamp):
        return self.result


def test_out_result_becomes_event():
    result = LivePipelineResult(
        frame_number=10,
        tracking_status="LOCKED",
        confidence=.99,
        decision="OUT",
        decision_confidence=.99,
    )

    adapter = RealTimeOfficiatingAdapter(
        FakePerception(result),
        minimum_call_confidence=.98,
    )

    output = adapter.process_frame(
        frame_number=10,
        frame=np.zeros((10,10,3), dtype=np.uint8),
        timestamp=1.0,
    )

    assert output.event is not None
    assert output.event.decision == LiveDecision.OUT


def test_low_confidence_call_becomes_review():
    result = LivePipelineResult(
        frame_number=10,
        tracking_status="LOCKED",
        confidence=.8,
        decision="IN",
        decision_confidence=.8,
    )

    adapter = RealTimeOfficiatingAdapter(
        FakePerception(result),
        minimum_call_confidence=.98,
    )

    output = adapter.process_frame(
        frame_number=10,
        frame=np.zeros((10,10,3), dtype=np.uint8),
        timestamp=1.0,
    )

    assert output.event.decision == LiveDecision.REVIEW
    assert output.event.metadata["original_decision"] == "IN"


def test_no_decision_means_no_event():
    result = LivePipelineResult(
        frame_number=1,
        tracking_status="SEARCHING",
    )

    adapter = RealTimeOfficiatingAdapter(
        FakePerception(result),
    )

    output = adapter.process_frame(
        frame_number=1,
        frame=np.zeros((10,10,3), dtype=np.uint8),
        timestamp=1.0,
    )

    assert output.event is None
