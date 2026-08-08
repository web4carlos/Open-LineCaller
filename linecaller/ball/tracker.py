import math
from .kalman import Kalman2D
from .models import BallTrackState, TrackStatus
from .trajectory import TrajectoryHistory, TrajectoryPoint

class BallTracker:
    def __init__(self, max_match_distance_px=120.0, max_lost_frames=5):
        self.max_match_distance_px = float(max_match_distance_px)
        self.max_lost_frames = int(max_lost_frames)
        self.kalman = Kalman2D()
        self.trajectory = TrajectoryHistory()
        self.lost_frames = 0

    def _select(self, candidates, predicted):
        if not candidates:
            return None
        if predicted is None:
            return max(candidates, key=lambda c: c.confidence)
        px, py = predicted
        valid = []
        for c in candidates:
            d = math.hypot(c.x-px, c.y-py)
            if d <= self.max_match_distance_px:
                valid.append((d - c.confidence*20.0, c))
        return min(valid, key=lambda t: t[0])[1] if valid else None

    def update(self, frame_number, candidates):
        predicted = self.kalman.predict() if self.kalman.initialized else None
        selected = self._select(candidates, predicted)

        if selected is not None:
            x, y = self.kalman.update(selected.x, selected.y)
            self.lost_frames = 0
            conf = max(0.0, min(1.0, selected.confidence))
            state = BallTrackState(frame_number, x, y, selected.confidence, conf, 0, TrackStatus.TRACKING)
            self.trajectory.add(TrajectoryPoint(frame_number, x, y, True, conf))
            return state

        self.lost_frames += 1
        if predicted is not None and self.lost_frames <= self.max_lost_frames:
            x, y = predicted
            persistence = max(0.0, 1.0 - self.lost_frames / max(1, self.max_lost_frames))
            conf = 0.65 * persistence
            state = BallTrackState(frame_number, x, y, 0.0, conf, self.lost_frames, TrackStatus.PREDICTED)
            self.trajectory.add(TrajectoryPoint(frame_number, x, y, False, conf))
            return state

        return BallTrackState(frame_number, None, None, 0.0, 0.0, self.lost_frames, TrackStatus.LOST)
