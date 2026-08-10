from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector
from linecaller.live.pipeline_factory import create_live_pipeline_adapter


def test_factory_uses_advanced_detector():
    adapter = create_live_pipeline_adapter()
    detector = adapter.perception.ball_engine.detector

    assert isinstance(detector, AdvancedMotionBallDetector)
