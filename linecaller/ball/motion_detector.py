from __future__ import annotations

import math
import cv2
import numpy as np

from linecaller.ball.detector import BallDetector
from linecaller.ball.models import BallCandidate


class MotionBallDetector(BallDetector):
    """
    Baseline candidate detector for integration testing.

    It intentionally returns multiple plausible small moving objects.
    BallTracker is responsible for temporal association.
    """

    def __init__(
        self,
        *,
        min_area: float = 8.0,
        max_area: float = 700.0,
        min_radius_px: float = 1.5,
        max_radius_px: float = 28.0,
        min_circularity: float = 0.20,
        history: int = 250,
        var_threshold: float = 20.0,
    ):
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)
        self.min_circularity = float(min_circularity)

        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray) -> list[BallCandidate]:
        mask = self.background.apply(frame)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[BallCandidate] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (self.min_area <= area <= self.max_area):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if not (self.min_radius_px <= radius <= self.max_radius_px):
                continue

            x0, y0, w, h = cv2.boundingRect(contour)
            aspect = min(w, h) / max(1.0, float(max(w, h)))

            roi = frame[y0:y0+h, x0:x0+w]
            brightness = 0.0
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                brightness = float(np.mean(hsv[..., 2])) / 255.0

            shape_score = max(0.0, min(1.0, circularity))
            confidence = (
                0.55 * shape_score
                + 0.25 * max(0.0, min(1.0, aspect))
                + 0.20 * brightness
            )

            candidates.append(
                BallCandidate(
                    x=float(x),
                    y=float(y),
                    radius_px=radius,
                    confidence=max(0.0, min(1.0, confidence)),
                    source="motion",
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
