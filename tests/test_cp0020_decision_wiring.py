from dataclasses import dataclass
import numpy as np

from linecaller.live.artificial_vision_perception import (
    RealArtificialVisionPerceptionAdapter,
)


class FakeStatus:
    value = "TRACKING"


@dataclass
class FakeState:
    frame_number: int
    x: float
    y: float
    tracking_confidence: float
    status: object


@dataclass
class FakeBounce:
    frame_number: int
    x: float
    y: float
    confidence: float


class DummyDetector:
    def detect(self, frame):
        return []


class FakeBallEngine:
    def process(self, frame_number, frame):
        return FakeState(
            frame_number,
            100.0,
            200.0,
            .95,
            FakeStatus(),
        )


class FakeBounceEngine:
    def process(self, point):
        return FakeBounce(
            10,
            101.0,
            201.0,
            .91,
        )


class FakeDecisionValue:
    value = "OUT"


class FakeDecisionResult:
    decision = FakeDecisionValue()
    confidence = .99
    explanation = ("test",)


class FakeDecisionEngine:
    def __init__(self):
        self.calls = 0

    def decide(self, bounce, calibration):
        self.calls += 1
        return FakeDecisionResult()


def test_calibration_enables_real_decision():
    decision_engine = FakeDecisionEngine()

    adapter = RealArtificialVisionPerceptionAdapter(
        detector=DummyDetector(),
        bounce_engine=FakeBounceEngine(),
        decision_engine=decision_engine,
        calibration=object(),
    )
    adapter.ball_engine = FakeBallEngine()

    result = adapter.process(
        frame_number=12,
        frame=np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
        timestamp=.5,
    )

    assert result.bounce_detected
    assert result.decision == "OUT"
    assert result.decision_confidence == .99
    assert decision_engine.calls == 1
