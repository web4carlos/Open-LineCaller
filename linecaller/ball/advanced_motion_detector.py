from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from linecaller.ball.detector import BallDetector
from linecaller.ball.models import BallCandidate
from linecaller.ball.detector_profile import DetectorProfile


@dataclass
class AdvancedDetectorTelemetry:
    frames: int = 0
    contours: int = 0
    area_rejected: int = 0
    shape_rejected: int = 0
    radius_rejected: int = 0
    confidence_rejected: int = 0
    accepted: int = 0


@dataclass(frozen=True)
class CandidateDiagnostic:
    x: float
    y: float
    radius_px: float
    area: float
    circularity: float
    aspect: float
    confidence: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class DetectorFrameAnalysis:
    mask: np.ndarray
    candidates: tuple[BallCandidate, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]


class AdvancedMotionBallDetector(BallDetector):
    def __init__(
        self,
        *,
        min_area=4.0,
        max_area=1400.0,
        min_radius_px=1.0,
        max_radius_px=35.0,
        min_circularity=0.08,
        history=300,
        var_threshold=16.0,
        max_candidates=24,
        temporal_radius_px=180.0,
        min_confidence=0.0,
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
        self.min_confidence = float(min_confidence)

        self.telemetry = AdvancedDetectorTelemetry()
        self._last_best = None
        self._make_background()

    @classmethod
    def from_profile(cls, profile):
        return cls(**profile.to_dict())

    def to_profile(self):
        return DetectorProfile(
            min_area=self.min_area,
            max_area=self.max_area,
            min_radius_px=self.min_radius_px,
            max_radius_px=self.max_radius_px,
            min_circularity=self.min_circularity,
            history=self.history,
            var_threshold=self.var_threshold,
            max_candidates=self.max_candidates,
            temporal_radius_px=self.temporal_radius_px,
            min_confidence=self.min_confidence,
        )

    def apply_profile(self, profile, reset=True):
        for key, value in profile.to_dict().items():
            setattr(self, key, value)

        if reset:
            self.reset()

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
        """
        Compatibility + scoring helper retained from CP-0021.

        OpenCV hue range is 0..179. This intentionally uses a broad
        fluorescent yellow/green band and weights it by saturation/value.
        """
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

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )

        return mask

    def analyze(self, frame):
        self.telemetry.frames += 1

        mask = self._foreground_mask(frame)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        self.telemetry.contours += len(contours)

        accepted = []
        diagnostics = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if not (self.min_area <= area <= self.max_area):
                self.telemetry.area_rejected += 1
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                self.telemetry.shape_rejected += 1
                continue

            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter)
            )

            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)

            x0, y0, width, height = cv2.boundingRect(contour)
            aspect = min(width, height) / max(
                1.0,
                float(max(width, height)),
            )

            roi = frame[y0:y0 + height, x0:x0 + width]

            color_affinity = 0.0
            brightness = 0.0
            saturation = 0.0

            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                color_affinity = self._yellow_green_affinity(hsv)
                brightness = float(np.mean(hsv[..., 2])) / 255.0
                saturation = float(np.mean(hsv[..., 1])) / 255.0

            temporal = self._temporal_score(x, y)

            confidence = self._clamp01(
                0.30 * self._clamp01(circularity)
                + 0.18 * self._clamp01(aspect)
                + 0.12 * brightness
                + 0.08 * saturation
                + 0.12 * color_affinity
                + 0.20 * temporal
            )

            reason = "ACCEPT"
            ok = True

            if circularity < self.min_circularity:
                self.telemetry.shape_rejected += 1
                reason = "CIRCULARITY"
                ok = False

            elif not (
                self.min_radius_px
                <= radius
                <= self.max_radius_px
            ):
                self.telemetry.radius_rejected += 1
                reason = "RADIUS"
                ok = False

            elif confidence < self.min_confidence:
                self.telemetry.confidence_rejected += 1
                reason = "CONFIDENCE"
                ok = False

            diagnostics.append(
                CandidateDiagnostic(
                    x=float(x),
                    y=float(y),
                    radius_px=radius,
                    area=area,
                    circularity=circularity,
                    aspect=aspect,
                    confidence=confidence,
                    accepted=ok,
                    reason=reason,
                )
            )

            if ok:
                accepted.append(
                    BallCandidate(
                        x=float(x),
                        y=float(y),
                        radius_px=radius,
                        confidence=confidence,
                        source="advanced-motion",
                    )
                )

        accepted.sort(
            key=lambda c: c.confidence,
            reverse=True,
        )
        accepted = accepted[: self.max_candidates]

        if accepted:
            self._last_best = (
                float(accepted[0].x),
                float(accepted[0].y),
            )
            self.telemetry.accepted += len(accepted)

        return DetectorFrameAnalysis(
            mask=mask,
            candidates=tuple(accepted),
            diagnostics=tuple(diagnostics),
        )

    def detect(self, frame):
        return list(self.analyze(frame).candidates)
