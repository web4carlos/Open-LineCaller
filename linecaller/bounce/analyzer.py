from dataclasses import dataclass
from statistics import mean
from linecaller.ball.trajectory import TrajectoryPoint

@dataclass(frozen=True)
class LocalTrajectoryAnalysis:
    center: TrajectoryPoint
    pre_vertical_velocity: float
    post_vertical_velocity: float
    measured_support_ratio: float
    mean_tracking_confidence: float
    direction_change_score: float

class TrajectoryAnalyzer:
    def __init__(self, half_window: int = 2):
        if half_window < 1:
            raise ValueError("half_window must be >= 1")
        self.half_window = half_window

    @staticmethod
    def _vy(a, b):
        df = b.frame_number - a.frame_number
        if df <= 0:
            raise ValueError("Trajectory frames must increase")
        return (b.y - a.y) / df

    def analyze(self, points, center_index):
        hw = self.half_window
        if center_index - hw < 0 or center_index + hw >= len(points):
            return None
        window = list(points[center_index-hw:center_index+hw+1])
        center = points[center_index]
        pre = mean(self._vy(window[i], window[i+1]) for i in range(hw))
        post = mean(self._vy(window[i], window[i+1]) for i in range(hw, len(window)-1))
        measured = sum(1 for p in window if p.measured) / len(window)
        conf = mean(max(0.0, min(1.0, p.confidence)) for p in window)
        score = max(0.0, pre) + max(0.0, -post)
        return LocalTrajectoryAnalysis(center, float(pre), float(post), float(measured), float(conf), float(score))
