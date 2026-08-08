from dataclasses import dataclass

@dataclass(frozen=True)
class BounceEvidence:
    pre_vertical_velocity: float
    post_vertical_velocity: float
    measured_support_ratio: float
    mean_tracking_confidence: float
    direction_change_score: float

@dataclass(frozen=True)
class BounceEvent:
    frame_number: int
    x: float
    y: float
    confidence: float
    evidence: BounceEvidence
