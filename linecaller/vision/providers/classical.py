from linecaller.ball.advanced_motion_detector import AdvancedMotionBallDetector
from linecaller.ball.engine import BallEngine
from linecaller.ball.tracker import BallTracker
from linecaller.vision.models import VisionDetection

class ClassicalVisionProvider:
    def __init__(self, *, detector=None, tracker=None):
        self._detector = detector or AdvancedMotionBallDetector()
        self._tracker = tracker or BallTracker()
        self._engine = BallEngine(self._detector, self._tracker)
        self._initialized = False

    @property
    def name(self):
        return "classical"

    def initialize(self):
        self._initialized = True

    def detect(self, *, frame, frame_number, timestamp):
        if not self._initialized:
            self.initialize()
        state = self._engine.process(frame_number, frame)
        status = getattr(state, "status", None)
        status = str(getattr(status, "value", status or "SEARCHING")).upper()
        return VisionDetection(
            frame_number=int(frame_number),
            x=getattr(state, "x", None),
            y=getattr(state, "y", None),
            confidence=float(getattr(state, "tracking_confidence", 0.0) or 0.0),
            status=status,
            source=self.name,
            metadata={
                "timestamp": float(timestamp),
                "detector": type(self._detector).__name__,
                "tracker": type(self._tracker).__name__,
            },
        )

    def reset(self):
        reset = getattr(self._detector, "reset", None)
        if callable(reset):
            reset()
        self._tracker = BallTracker()
        self._engine = BallEngine(self._detector, self._tracker)

    def shutdown(self):
        self._initialized = False
