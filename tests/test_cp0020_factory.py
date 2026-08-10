import pytest

from linecaller.live.artificial_vision_perception import (
    RealArtificialVisionPerceptionAdapter,
)
from linecaller.live.pipeline_factory import (
    create_live_pipeline_adapter,
)


class DummyDetector:
    def detect(self, frame):
        return []


def test_factory_default_is_real_perception():
    adapter = create_live_pipeline_adapter(
        detector=DummyDetector()
    )

    assert (
        type(adapter.perception)
        is RealArtificialVisionPerceptionAdapter
    )


def test_factory_preserves_minimum_confidence():
    adapter = create_live_pipeline_adapter(
        detector=DummyDetector(),
        minimum_call_confidence=.87,
    )

    assert (
        adapter.minimum_call_confidence
        == pytest.approx(.87)
    )


def test_factory_rejects_two_calibration_sources():
    with pytest.raises(ValueError):
        create_live_pipeline_adapter(
            detector=DummyDetector(),
            calibration=object(),
            calibration_path="court.json",
        )
