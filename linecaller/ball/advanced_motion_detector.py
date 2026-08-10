from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from linecaller.ball.detector import BallDetector
from linecaller.ball.models import BallCandidate


@dataclass
class AdvancedDetectorTelemetry:
    frames: int = 0
    contours: int = 0
    area_rejected: int = 0
    shape_rejected: int = 0
    radius_rejected: int = 0
    accepted: int = 0


class AdvancedMotionBallDetector(BallDetector):
    def __init__(
        self,
        *,
        min_area: float = 4.0,
        max_area: float = 1400.0,
        min_radius_px: float = 1.0,
        max_radius_px: float = 35.0,
        min_circularity: float = 0.08,
        history: int = 300,
        var_threshold: float = 16.0,
        max_candidates: int = 24,
        temporal_radius_px: float = 180.0,
    ):
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)
        self.min_circularity = float(min_circularity)
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.max_candidates = int(max_candidates)
        self.temporal_radius_px = float(temporal_radius_px)

        self.telemetry = AdvancedDetectorTelemetry()
        self._last_best = None
        self._make_background()

    def _make_background(self):
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )

    def reset(self):
        self._make_background()
        self._last_best = None
        self.telemetry = AdvancedDetectorTelemetry()

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _yellow_green_affinity(hsv_roi):
        if hsv_roi.size == 0:
            return 0.0

        h = hsv_roi[..., 0].astype(np.float32)
        s = hsv_roi[..., 1].astype(np.float32) / 255.0
        v = hsv_roi[..., 2].astype(np.float32) / 255.0

        hue_mask = (h >= 20.0) & (h <= 55.0)
        quality = hue_mask.astype(np.float32) * s * v
        return float(np.mean(quality))

    def _temporal_score(self, x, y):
        if self._last_best is None:
            return 0.5

        lx, ly = self._last_best
        d = math.hypot(float(x) - lx, float(y) - ly)

        if d >= self.temporal_radius_px:
            return 0.0

        return 1.0 - d / max(1.0, self.temporal_radius_px)

    def _foreground_mask(self, frame):
        mask = self.background.apply(frame)
        mask = cv2.medianBlur(mask, 3)

        open_kernel = np.ones((2, 2), dtype=np.uint8)
        close_kernel = np.ones((5, 5), dtype=np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def detect(self, frame: np.ndarray) -> list[BallCandidate]:
        self.telemetry.frames += 1

        mask = self._foreground_mask(frame)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        self.telemetry.contours += len(contours)
        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if not (self.min_area <= area <= self.max_area):
                self.telemetry.area_rejected += 1
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                self.telemetry.shape_rejected += 1
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)

            if circularity < self.min_circularity:
                self.telemetry.shape_rejected += 1
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)

            if not (self.min_radius_px <= radius <= self.max_radius_px):
                self.telemetry.radius_rejected += 1
                continue

            x0, y0, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                self.telemetry.shape_rejected += 1
                continue

            aspect = min(width, height) / max(1.0, float(max(width, height)))

            roi = frame[y0:y0 + height, x0:x0 + width]
            brightness = saturation = color_affinity = 0.0

            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                brightness = float(np.mean(hsv[..., 2])) / 255.0
                saturation = float(np.mean(hsv[..., 1])) / 255.0
                color_affinity = self._yellow_green_affinity(hsv)

            temporal_score = self._temporal_score(x, y)
            size_mid = (self.min_radius_px + self.max_radius_px) / 2.0
            size_span = max(
                1.0,
                (self.max_radius_px - self.min_radius_px) / 2.0,
            )
            size_score = self._clamp01(
                1.0 - abs(radius - size_mid) / size_span
            )

            confidence = self._clamp01(
                0.25 * self._clamp01(circularity)
                + 0.15 * self._clamp01(aspect)
                + 0.12 * brightness
                + 0.08 * saturation
                + 0.15 * color_affinity
                + 0.20 * temporal_score
                + 0.05 * size_score
            )

            candidates.append(
                BallCandidate(
                    x=float(x),
                    y=float(y),
                    radius_px=radius,
                    confidence=confidence,
                    source="advanced-motion",
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        candidates = candidates[: self.max_candidates]

        if candidates:
            best = candidates[0]
            self._last_best = (float(best.x), float(best.y))
            self.telemetry.accepted += len(candidates)

        return candidates
