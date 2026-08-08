from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float
    angle_deg: float


class AutoCalibrationDisposition(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    AUTO_REVIEW = "AUTO_REVIEW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class AutoCalibrationProposal:
    corners: tuple[tuple[float, float], ...]
    confidence: float
    disposition: AutoCalibrationDisposition
    line_count: int
    family_a_count: int
    family_b_count: int
    orientation_separation_deg: float
    quadrilateral_area_ratio: float
    reasons: tuple[str, ...]
