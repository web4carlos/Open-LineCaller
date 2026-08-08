from .tracker import BallTracker

class BallEngine:
    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker or BallTracker()

    def process(self, frame_number, frame):
        candidates = self.detector.detect(frame)
        return self.tracker.update(frame_number, candidates)
