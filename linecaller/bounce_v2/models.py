from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MotionSample:
    frame: int
    raw_x: Optional[float]
    raw_y: Optional[float]
    tracked_x: Optional[float]
    tracked_y: Optional[float]
    confidence: float
    source: str

@dataclass(frozen=True)
class BounceEvent:
    frame: int
    x: float
    y: float
    confidence: float
    pre_velocity_y: float
    post_velocity_y: float
    source: str
    curvature: float = 0.0
    bounce_score: float = 0.0
