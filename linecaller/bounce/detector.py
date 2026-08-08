from dataclasses import dataclass
from .analyzer import TrajectoryAnalyzer
from .models import BounceEvent, BounceEvidence

@dataclass(frozen=True)
class BounceThresholds:
    min_downward_speed_px_per_frame: float = 2.0
    min_upward_speed_px_per_frame: float = 2.0
    min_measured_support_ratio: float = 0.60
    min_tracking_confidence: float = 0.45
    strong_direction_change_px_per_frame: float = 12.0

class BounceDetector:
    def __init__(self, analyzer=None, thresholds=None):
        self.analyzer = analyzer or TrajectoryAnalyzer(2)
        self.thresholds = thresholds or BounceThresholds()

    def detect_at(self, points, center_index):
        a = self.analyzer.analyze(points, center_index)
        if a is None:
            return None
        t = self.thresholds
        if a.pre_vertical_velocity < t.min_downward_speed_px_per_frame:
            return None
        if a.post_vertical_velocity > -t.min_upward_speed_px_per_frame:
            return None
        if a.measured_support_ratio < t.min_measured_support_ratio:
            return None
        if a.mean_tracking_confidence < t.min_tracking_confidence:
            return None

        direction = min(1.0, a.direction_change_score / max(1e-9, t.strong_direction_change_px_per_frame))
        confidence = max(0.0, min(1.0,
            0.50 * direction +
            0.25 * a.measured_support_ratio +
            0.25 * a.mean_tracking_confidence
        ))
        return BounceEvent(
            frame_number=a.center.frame_number,
            x=a.center.x,
            y=a.center.y,
            confidence=confidence,
            evidence=BounceEvidence(
                a.pre_vertical_velocity,
                a.post_vertical_velocity,
                a.measured_support_ratio,
                a.mean_tracking_confidence,
                a.direction_change_score,
            ),
        )
