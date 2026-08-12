from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DetectionPoint:
    frame: int
    x: float
    y: float
    confidence: float

@dataclass(frozen=True)
class TrackPoint:
    frame: int
    raw_x: Optional[float]
    raw_y: Optional[float]
    tracked_x: Optional[float]
    tracked_y: Optional[float]
    confidence: float
    source: str
    missed_frames: int
