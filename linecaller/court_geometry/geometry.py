from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .calibration import CourtCalibration


@dataclass(frozen=True)
class CourtPointResult:
    image_x: float
    image_y: float
    court_x_ft: float
    court_y_ft: float
    inside_court: bool
    nearest_line: str
    signed_distance_ft: float
    absolute_distance_ft: float
    geometry_state: str


class CourtGeometry:
    """
    Perspective mapping from image pixels to a 20x44 ft pickleball court.

    This module reports geometry only.
    CP-0029 will convert geometry + bounce confidence into final IN/OUT/REVIEW.
    """

    def __init__(
        self,
        calibration: CourtCalibration,
        *,
        near_line_threshold_in: float = 3.0,
    ):
        self.calibration = calibration
        self.near_line_threshold_ft = (
            float(near_line_threshold_in) / 12.0
        )

        src = np.array(
            calibration.image_points,
            dtype=np.float32,
        )

        w = calibration.court_width_ft
        l = calibration.court_length_ft

        dst = np.array(
            [
                [0.0, l],
                [w, l],
                [w, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.H = cv2.getPerspectiveTransform(src, dst)

    def image_to_court(self, x, y):
        src = np.array([[[float(x), float(y)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def classify(self, image_x, image_y):
        x, y = self.image_to_court(image_x, image_y)

        w = self.calibration.court_width_ft
        l = self.calibration.court_length_ft

        inside = (
            0.0 <= x <= w
            and 0.0 <= y <= l
        )

        distances = {
            "LEFT_SIDELINE": x,
            "RIGHT_SIDELINE": w - x,
            "FAR_BASELINE": y,
            "NEAR_BASELINE": l - y,
        }

        nearest_line = min(
            distances,
            key=lambda k: abs(distances[k]),
        )
        signed = float(distances[nearest_line])
        absolute = abs(signed)

        if absolute <= self.near_line_threshold_ft:
            state = "NEAR_LINE"
        elif inside:
            state = "INSIDE"
        else:
            state = "OUTSIDE"

        return CourtPointResult(
            image_x=float(image_x),
            image_y=float(image_y),
            court_x_ft=x,
            court_y_ft=y,
            inside_court=inside,
            nearest_line=nearest_line,
            signed_distance_ft=signed,
            absolute_distance_ft=absolute,
            geometry_state=state,
        )
