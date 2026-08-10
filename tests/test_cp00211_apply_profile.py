from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector
from linecaller.ball.detector_profile import DetectorProfile

def test_apply_profile():
    d=AdvancedMotionBallDetector()
    p=DetectorProfile(min_circularity=.4,min_confidence=.6,max_candidates=4)
    d.apply_profile(p)
    assert d.min_circularity == .4
    assert d.min_confidence == .6
    assert d.max_candidates == 4
