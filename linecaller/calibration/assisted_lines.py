from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class LineCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float
    angle_deg: float


class AssistedLineDetector:
    """
    Candidate generator only.

    Produces long straight-line candidates. It does not yet identify
    baseline/sideline/NVZ/centerline semantics.
    """

    def __init__(
        self,
        *,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 70,
        min_line_length: int = 80,
        max_line_gap: int = 20,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, frame: np.ndarray) -> list[LineCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        candidates: list[LineCandidate] = []
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, raw)
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            candidates.append(
                LineCandidate(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    length_px=length, angle_deg=angle
                )
            )

        candidates.sort(key=lambda item: item.length_px, reverse=True)
        return candidates
