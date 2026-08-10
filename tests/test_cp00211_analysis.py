import numpy as np
from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector

def test_analysis_returns_mask_candidates_diagnostics():
    d=AdvancedMotionBallDetector()
    frame=np.zeros((80,120,3),dtype=np.uint8)
    a=d.analyze(frame)
    assert a.mask.shape == frame.shape[:2]
    assert isinstance(a.candidates,tuple)
    assert isinstance(a.diagnostics,tuple)

def test_confidence_threshold_is_profiled():
    d=AdvancedMotionBallDetector(min_confidence=.55)
    assert d.to_profile().min_confidence == .55
