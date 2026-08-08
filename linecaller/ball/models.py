from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    radius_px: float
    confidence: float
    source: str = "unknown"

class TrackStatus(str, Enum):
    TRACKING = "TRACKING"
    PREDICTED = "PREDICTED"
    LOST = "LOST"

@dataclass(frozen=True)
class BallTrackState:
    frame_number: int
    x: float | None
    y: float | None
    detector_confidence: float
    tracking_confidence: float
    lost_frames: int
    status: TrackStatus
