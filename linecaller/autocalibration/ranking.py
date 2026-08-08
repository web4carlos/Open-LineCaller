from __future__ import annotations
from dataclasses import dataclass

from .hypothesis import CourtHypothesis
from .models import AutoCalibrationDisposition


@dataclass(frozen=True)
class RankedAutoCalibration:
    best: CourtHypothesis | None
    runner_up: CourtHypothesis | None
    confidence_margin: float
    disposition: AutoCalibrationDisposition
    reasons: tuple[str, ...]


class HypothesisRanker:
    def __init__(
        self,
        *,
        accept_score: float = 0.72,
        review_score: float = 0.45,
        min_accept_margin: float = 0.08,
        min_review_margin: float = 0.03,
    ):
        self.accept_score = float(accept_score)
        self.review_score = float(review_score)
        self.min_accept_margin = float(min_accept_margin)
        self.min_review_margin = float(min_review_margin)

    def rank(self, hypotheses: list[CourtHypothesis]) -> RankedAutoCalibration:
        if not hypotheses:
            return RankedAutoCalibration(
                best=None,
                runner_up=None,
                confidence_margin=0.0,
                disposition=AutoCalibrationDisposition.REJECT,
                reasons=("No court hypotheses generated",),
            )

        best = hypotheses[0]
        runner = hypotheses[1] if len(hypotheses) > 1 else None
        margin = best.score - (runner.score if runner else 0.0)

        reasons = []

        if best.score >= self.accept_score and margin >= self.min_accept_margin:
            disposition = AutoCalibrationDisposition.AUTO_ACCEPT
        elif best.score >= self.review_score and margin >= self.min_review_margin:
            disposition = AutoCalibrationDisposition.AUTO_REVIEW
        else:
            disposition = AutoCalibrationDisposition.REJECT

            if best.score < self.review_score:
                reasons.append("Best hypothesis score below review threshold")

            if runner is not None and margin < self.min_review_margin:
                reasons.append("Top hypotheses are too ambiguous")

        return RankedAutoCalibration(
            best=best,
            runner_up=runner,
            confidence_margin=float(margin),
            disposition=disposition,
            reasons=tuple(reasons),
        )
