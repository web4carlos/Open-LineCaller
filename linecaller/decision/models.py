from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    IN = "IN"
    OUT = "OUT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class DecisionContext:
    bounce_frame: int
    image_x: float
    image_y: float

    court_x_m: float | None
    court_y_m: float | None

    nearest_line: str | None
    signed_distance_m: float | None

    local_m_per_px: float | None
    calibration_error_px: float | None
    bounce_error_px: float
    total_uncertainty_m: float | None

    bounce_confidence: float
    calibration_valid: bool

    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    confidence: float
    context: DecisionContext
    explanation: tuple[str, ...]
