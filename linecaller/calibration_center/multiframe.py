from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

import cv2
import numpy as np

from .auto import AutoCourtCalibrator


@dataclass(frozen=True)
class LineConfidence:
    name: str
    confidence: float
    status: str


@dataclass(frozen=True)
class MultiFrameCalibrationResult:
    success: bool
    confidence: float
    image_points: tuple[tuple[float, float], ...] | None
    line_confidences: tuple[LineConfidence, ...]
    frames_analyzed: int
    frames_with_candidate: int
    reason: str


class MultiFrameCourtCalibrator:
    """
    CP-0030.7 Multi-Frame Calibration Evidence.

    Strategy:
    - run AUTO on multiple frames;
    - collect valid court quadrilaterals;
    - reject geometric outliers;
    - aggregate corner positions robustly with medians;
    - measure temporal support for each physical court boundary;
    - expose per-line confidence;
    - require enough line support before AUTO is considered trustworthy.
    """

    LINE_NAMES = (
        "NEAR_BASELINE",
        "RIGHT_SIDELINE",
        "FAR_BASELINE",
        "LEFT_SIDELINE",
    )

    def __init__(
        self,
        *,
        min_auto_confidence: float = 0.45,
        min_frame_support: float = 0.30,
        min_line_confidence: float = 0.45,
        review_line_confidence: float = 0.65,
    ):
        self.auto = AutoCourtCalibrator(
            min_confidence=min_auto_confidence
        )
        self.min_frame_support = float(min_frame_support)
        self.min_line_confidence = float(min_line_confidence)
        self.review_line_confidence = float(review_line_confidence)

    @staticmethod
    def _quad_array(points):
        return np.asarray(points, dtype=np.float64).reshape(4, 2)

    @staticmethod
    def _quad_center(q):
        return q.mean(axis=0)

    @staticmethod
    def _quad_scale(q):
        return max(
            1.0,
            np.linalg.norm(q[1] - q[0])
            + np.linalg.norm(q[2] - q[3]),
        )

    def _reject_outliers(self, quads):
        if len(quads) <= 2:
            return quads

        centers = np.array(
            [self._quad_center(q) for q in quads],
            dtype=float,
        )
        med_center = np.median(centers, axis=0)

        distances = np.linalg.norm(
            centers - med_center,
            axis=1,
        )

        med_d = float(np.median(distances))
        mad = float(
            np.median(
                np.abs(
                    distances - med_d
                )
            )
        )

        threshold = max(
            12.0,
            med_d + 3.5 * max(mad, 1.0),
        )

        kept = [
            q
            for q, d in zip(quads, distances)
            if d <= threshold
        ]

        return kept or quads

    @staticmethod
    def _aggregate(quads):
        stack = np.stack(quads, axis=0)
        med = np.median(stack, axis=0)
        return tuple(
            (float(x), float(y))
            for x, y in med
        )

    @staticmethod
    def _line_motion_confidence(quads, i, j):
        """
        Confidence from temporal stability of each boundary.
        Smaller midpoint variance -> higher confidence.
        """
        mids = []
        lengths = []

        for q in quads:
            a = q[i]
            b = q[j]
            mids.append((a + b) / 2.0)
            lengths.append(
                float(
                    np.linalg.norm(
                        b - a
                    )
                )
            )

        mids = np.asarray(mids, dtype=float)

        if len(mids) <= 1:
            return 0.35

        spread = np.median(
            np.linalg.norm(
                mids - np.median(
                    mids,
                    axis=0,
                ),
                axis=1,
            )
        )

        typical_length = max(
            1.0,
            float(
                median(lengths)
            ),
        )

        relative = spread / typical_length

        return float(
            np.clip(
                1.0 - relative * 8.0,
                0.0,
                1.0,
            )
        )

    def calibrate_frames(
        self,
        frames: Iterable[np.ndarray],
    ):
        total = 0
        candidates = []
        source_confidences = []

        for frame in frames:
            total += 1

            result = self.auto.detect(
                frame
            )

            if (
                result.calibration
                is None
            ):
                continue

            candidates.append(
                self._quad_array(
                    result.calibration.image_points
                )
            )

            source_confidences.append(
                float(
                    result.confidence
                )
            )

        if total == 0:
            return MultiFrameCalibrationResult(
                False,
                0.0,
                None,
                (),
                0,
                0,
                "NO_FRAMES",
            )

        if not candidates:
            return MultiFrameCalibrationResult(
                False,
                0.0,
                None,
                (),
                total,
                0,
                "NO_AUTO_CANDIDATES",
            )

        filtered = self._reject_outliers(
            candidates
        )

        frame_support = (
            len(filtered)
            / total
        )

        if frame_support < self.min_frame_support:
            return MultiFrameCalibrationResult(
                False,
                frame_support,
                None,
                (),
                total,
                len(filtered),
                "INSUFFICIENT_TEMPORAL_SUPPORT",
            )

        aggregate = self._aggregate(
            filtered
        )

        boundary_indices = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        )

        line_results = []

        for name, (i, j) in zip(
            self.LINE_NAMES,
            boundary_indices,
        ):
            stability = (
                self._line_motion_confidence(
                    filtered,
                    i,
                    j,
                )
            )

            confidence = float(
                np.clip(
                    0.65 * stability
                    + 0.35 * frame_support,
                    0.0,
                    1.0,
                )
            )

            if confidence < self.min_line_confidence:
                status = "LOW"
            elif confidence < self.review_line_confidence:
                status = "REVIEW"
            else:
                status = "GOOD"

            line_results.append(
                LineConfidence(
                    name,
                    confidence,
                    status,
                )
            )

        mean_auto = (
            float(
                np.mean(
                    source_confidences
                )
            )
            if source_confidences
            else 0.0
        )

        mean_line = float(
            np.mean(
                [
                    x.confidence
                    for x in line_results
                ]
            )
        )

        confidence = float(
            np.clip(
                0.45 * mean_auto
                + 0.35 * mean_line
                + 0.20 * frame_support,
                0.0,
                1.0,
            )
        )

        low_lines = [
            x
            for x in line_results
            if x.status == "LOW"
        ]

        success = not low_lines

        return MultiFrameCalibrationResult(
            success=success,
            confidence=confidence,
            image_points=aggregate,
            line_confidences=tuple(
                line_results
            ),
            frames_analyzed=total,
            frames_with_candidate=len(filtered),
            reason=(
                "OK"
                if success
                else "LOW_LINE_CONFIDENCE"
            ),
        )
