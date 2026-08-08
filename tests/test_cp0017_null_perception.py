import numpy as np

from linecaller.live.null_perception import NullLivePerceptionAdapter


def test_null_perception_is_honest():
    result = NullLivePerceptionAdapter().process(
        frame_number=3,
        frame=np.zeros((10,10,3), dtype=np.uint8),
        timestamp=0.0,
    )

    assert result.tracking_status == "SEARCHING"
    assert result.decision is None
    assert result.confidence == 0.0
