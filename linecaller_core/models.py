from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Call(str, Enum):
    IN = 'IN'
    OUT = 'OUT'
    REVIEW = 'REVIEW'


@dataclass(frozen=True)
class BallObservation:
    frame_number: int
    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True)
class BounceEvent:
    frame_number: int
    x_m: float
    y_m: float
    confidence: float


@dataclass(frozen=True)
class CourtModel:
    width_m: float = 6.096
    length_m: float = 13.4112
    line_width_m: float = 0.0508


@dataclass(frozen=True)
class DecisionResult:
    call: Call
    confidence: float
    signed_distance_m: float
    reason: str
