from __future__ import annotations

from dataclasses import dataclass

from linecaller.autocalibration.models import AutoCalibrationDisposition
from linecaller.autocalibration.ranking import RankedAutoCalibration


@dataclass(frozen=True)
class SmartCalibrationPrefill:
    accepted: bool
    disposition: AutoCalibrationDisposition
    outer_corners: tuple[tuple[float, float], ...]
    score: float
    margin: float
    reason: str


class SmartCalibrationAdapter:
    """
    Converts CP-0009 ranked output into safe CalibrationSession prefill data.

    Only outer corners are prefilled.
    """

    def from_ranked(
        self,
        ranked: RankedAutoCalibration,
    ) -> SmartCalibrationPrefill:

        if ranked.best is None:
            return SmartCalibrationPrefill(
                accepted=False,
                disposition=ranked.disposition,
                outer_corners=(),
                score=0.0,
                margin=ranked.confidence_margin,
                reason="No usable court hypothesis",
            )

        if ranked.disposition == AutoCalibrationDisposition.REJECT:
            return SmartCalibrationPrefill(
                accepted=False,
                disposition=ranked.disposition,
                outer_corners=ranked.best.corners,
                score=ranked.best.score,
                margin=ranked.confidence_margin,
                reason="Best hypothesis rejected by ambiguity/score policy",
            )

        return SmartCalibrationPrefill(
            accepted=True,
            disposition=ranked.disposition,
            outer_corners=ranked.best.corners,
            score=ranked.best.score,
            margin=ranked.confidence_margin,
            reason="Outer court proposal available",
        )
