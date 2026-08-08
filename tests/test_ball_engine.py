import numpy as np
from linecaller.ball.detector import BallDetector
from linecaller.ball.engine import BallEngine
from linecaller.ball.models import BallCandidate, TrackStatus

class MockDetector(BallDetector):
    def detect(self, frame):
        return [BallCandidate(50.0,60.0,4.0,0.95,"mock")]

def test_engine_tracks():
    engine = BallEngine(MockDetector())
    frame = np.zeros((100,100,3), dtype=np.uint8)
    state = engine.process(1, frame)
    assert state.status == TrackStatus.TRACKING
    assert state.detector_confidence == 0.95
