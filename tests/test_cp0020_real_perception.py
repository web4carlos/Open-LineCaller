from dataclasses import dataclass
import numpy as np

from linecaller.validation.annotation_models import (
    AnnotationCandidate,
)
from linecaller.live.artificial_vision_perception import (
    RealArtificialVisionPerceptionAdapter,
)


class FakeStatus:
    value = "TRACKING"


@dataclass
class FakeState:
    frame_number: int
    x: float | None
    y: float | None
    tracking_confidence: float
    status: object


class FakeBallEngine:
    def process(self, frame_number, frame):
        return FakeState(
            frame_number=frame_number,
            x=20.0,
            y=30.0,
            tracking_confidence=.9,
            status=FakeStatus(),
        )


@dataclass
class FakeBounce:
    frame_number: int = 7
    x: float = 21.0
    y: float = 31.0
    confidence: float = .88


class FakeBounceEngine:
    def __init__(self, event=None):
        self.event = event

    def process(self, point):
        return self.event


class DummyDetector:
    def detect(self, frame):
        return []


def make_adapter(bounce=None):
    adapter = RealArtificialVisionPerceptionAdapter(
        detector=DummyDetector(),
        bounce_engine=FakeBounceEngine(
            bounce
        ),
    )
    adapter.ball_engine = FakeBallEngine()
    return adapter


def test_tracking_result_without_bounce():
    adapter = make_adapter()

    result = adapter.process(
        frame_number=5,
        frame=np.zeros(
            (20, 20, 3),
            dtype=np.uint8,
        ),
        timestamp=.1,
    )

    assert result.tracking_status == "TRACKING"
    assert result.ball_x == 20.0
    assert result.ball_y == 30.0
    assert result.bounce_detected is False
    assert result.decision is None


def test_bounce_without_calibration_becomes_review():
    adapter = make_adapter(
        FakeBounce()
    )

    result = adapter.process(
        frame_number=9,
        frame=np.zeros(
            (20, 20, 3),
            dtype=np.uint8,
        ),
        timestamp=.2,
    )

    assert result.bounce_detected is True
    assert result.decision == "REVIEW"
    assert result.frame_number == 7
    assert (
        result.metadata["reason"]
        == "calibration_required"
    )
