import numpy as np
from linecaller.vision.providers.classical import ClassicalVisionProvider

class DummyDetector:
    def detect(self, frame):
        return []

def test_contract():
    p = ClassicalVisionProvider(detector=DummyDetector())
    frame = np.zeros((64,64,3), dtype=np.uint8)
    result = p.detect(frame=frame, frame_number=0, timestamp=0.0)
    assert result.frame_number == 0
    assert result.source == "classical"
    assert result.status
