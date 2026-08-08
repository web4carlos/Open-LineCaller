from __future__ import annotations

import math
import cv2
import numpy as np

from .models import LineSegment


def normalized_angle_deg(x1, y1, x2, y2) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


class HoughCourtLineDetector:
    def __init__(
        self,
        *,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 55,
        min_line_length: int = 70,
        max_line_gap: int = 25,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, frame: np.ndarray) -> list[LineSegment]:
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            self.canny_low,
            self.canny_high,
        )

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

        # OpenCV builds may return either:
        #   (N, 1, 4)
        # or
        #   (N, 4)
        # Normalize both to (N, 4).
        normalized = np.asarray(lines).reshape(-1, 4)

        result: list[LineSegment] = []

        for raw in normalized:
            x1, y1, x2, y2 = map(float, raw)

            length = math.hypot(
                x2 - x1,
                y2 - y1,
            )

            result.append(
                LineSegment(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_px=length,
                    angle_deg=normalized_angle_deg(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )
            )

        result.sort(
            key=lambda segment: segment.length_px,
            reverse=True,
        )

        return result
