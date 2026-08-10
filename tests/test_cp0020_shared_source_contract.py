import numpy as np

from linecaller.live.pipeline_factory import (
    create_live_pipeline_adapter,
)


class DummyDetector:
    def detect(self, frame):
        return []


def test_same_adapter_accepts_arbitrary_frame_source():
    adapter = create_live_pipeline_adapter(
        detector=DummyDetector(),
        minimum_call_confidence=0.0,
    )

    recorded_frame = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )
    live_camera_frame = np.ones(
        (32, 32, 3),
        dtype=np.uint8,
    )

    a = adapter.process_frame(
        frame_number=0,
        frame=recorded_frame,
        timestamp=0.0,
    )

    b = adapter.process_frame(
        frame_number=1,
        frame=live_camera_frame,
        timestamp=1 / 60.0,
    )

    assert a.result.frame_number == 0
    assert b.result.frame_number == 1
