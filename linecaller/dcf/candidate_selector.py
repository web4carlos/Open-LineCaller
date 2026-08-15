from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class BallCandidateObservation:
    frame: int
    x: float
    y: float
    confidence: float
    width: float
    height: float
    class_id: int | None = None


@dataclass(frozen=True)
class CandidateScore:
    candidate: BallCandidateObservation
    score: float
    distance_px: float
    size_ratio: float
    confidence_score: float
    continuity_score: float
    size_score: float


@dataclass(frozen=True)
class CandidateSelection:
    frame: int
    selected: BallCandidateObservation | None
    reason: str
    candidate_count: int
    scored: tuple[CandidateScore, ...]


class DCFBallCandidateSelector:
    """
    CP-0035.6
    Selects the most physically plausible YOLO candidate BEFORE Kalman lock.

    Current version uses:
      - predicted image position
      - candidate distance to prediction
      - apparent size continuity
      - confidence

    Future DCF versions will replace/augment image-space radius with
    illuminated 3-D field cells.
    """

    def __init__(
        self,
        *,
        max_distance_px: float = 85.0,
        max_size_ratio: float = 3.0,
        min_confidence: float = 0.05,
        confidence_weight: float = 0.25,
        continuity_weight: float = 0.55,
        size_weight: float = 0.20,
    ):
        self.max_distance_px = float(max_distance_px)
        self.max_size_ratio = float(max_size_ratio)
        self.min_confidence = float(min_confidence)
        self.confidence_weight = float(confidence_weight)
        self.continuity_weight = float(continuity_weight)
        self.size_weight = float(size_weight)

    @staticmethod
    def _size(c: BallCandidateObservation) -> float:
        return max(1.0, (float(c.width) + float(c.height)) / 2.0)

    def _size_ratio(self, candidate, expected_size):
        if expected_size is None:
            return 1.0
        a = self._size(candidate)
        b = max(1.0, float(expected_size))
        return max(a / b, b / a)

    def score_candidates(
        self,
        candidates: Iterable[BallCandidateObservation],
        *,
        expected_x: float | None,
        expected_y: float | None,
        expected_size: float | None = None,
    ) -> tuple[CandidateScore, ...]:
        scores = []

        for c in candidates:
            if c.confidence < self.min_confidence:
                continue

            if expected_x is None or expected_y is None:
                distance = 0.0
                continuity_score = 1.0
            else:
                distance = hypot(c.x - expected_x, c.y - expected_y)
                if distance > self.max_distance_px:
                    continue
                continuity_score = max(
                    0.0,
                    1.0 - distance / max(1e-9, self.max_distance_px),
                )

            size_ratio = self._size_ratio(c, expected_size)
            if size_ratio > self.max_size_ratio:
                continue

            size_score = 1.0 if expected_size is None else max(
                0.0,
                1.0 - (size_ratio - 1.0) / max(1e-9, self.max_size_ratio - 1.0),
            )

            conf_score = max(0.0, min(1.0, c.confidence))

            score = (
                self.continuity_weight * continuity_score
                + self.size_weight * size_score
                + self.confidence_weight * conf_score
            )

            scores.append(
                CandidateScore(
                    candidate=c,
                    score=score,
                    distance_px=distance,
                    size_ratio=size_ratio,
                    confidence_score=conf_score,
                    continuity_score=continuity_score,
                    size_score=size_score,
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        return tuple(scores)

    def select(
        self,
        candidates: Iterable[BallCandidateObservation],
        *,
        expected_x: float | None,
        expected_y: float | None,
        expected_size: float | None = None,
    ) -> CandidateSelection:
        candidates = tuple(candidates)
        scored = self.score_candidates(
            candidates,
            expected_x=expected_x,
            expected_y=expected_y,
            expected_size=expected_size,
        )

        if not candidates:
            return CandidateSelection(
                frame=-1,
                selected=None,
                reason="NO_CANDIDATES",
                candidate_count=0,
                scored=(),
            )

        frame = candidates[0].frame

        if not scored:
            return CandidateSelection(
                frame=frame,
                selected=None,
                reason="NO_PLAUSIBLE_CANDIDATE",
                candidate_count=len(candidates),
                scored=(),
            )

        return CandidateSelection(
            frame=frame,
            selected=scored[0].candidate,
            reason="DCF_PLAUSIBLE_CANDIDATE",
            candidate_count=len(candidates),
            scored=scored,
        )
