import numpy as np

from linecaller.live.pipeline_factory import create_live_pipeline_adapter


def test_detector_processes_sequential_live_frames():
    adapter = create_live_pipeline_adapter(
        minimum_call_confidence=0.0
    )

    f1 = np.zeros((64, 64, 3), dtype=np.uint8)
    f2 = f1.copy()
    f2[20:26, 20:26] = (0, 255, 255)

    out1 = adapter.process_frame(
        frame_number=0,
        frame=f1,
        timestamp=0.0,
    )

    out2 = adapter.process_frame(
        frame_number=1,
        frame=f2,
        timestamp=1 / 60.0,
    )

    assert out1.result.frame_number == 0
    assert out2.result.frame_number == 1
