from __future__ import annotations

from linecaller.bounce.models import BounceEvent
from linecaller.calibration.profile import CalibrationProfile

from .fusion import CourtFusion
from .models import Decision, DecisionContext, DecisionResult


class DecisionEngine:
    def __init__(
        self,
        *,
        fusion: CourtFusion | None = None,
        min_bounce_confidence: float = 0.70,
    ):
        self.fusion = fusion or CourtFusion()
        self.min_bounce_confidence = float(min_bounce_confidence)

    @staticmethod
    def _confidence_from_margin(
        distance_m: float,
        uncertainty_m: float,
        bounce_confidence: float,
    ) -> float:
        """
        Confidence grows as the signed-distance magnitude separates from the
        uncertainty band. It is intentionally capped by bounce confidence.
        """
        if uncertainty_m <= 0:
            geometry_conf = 1.0
        else:
            ratio = abs(distance_m) / uncertainty_m
            geometry_conf = max(0.0, min(1.0, (ratio - 1.0) / 4.0))

        return max(
            0.0,
            min(1.0, 0.5 * geometry_conf + 0.5 * bounce_confidence),
        )

    def decide(
        self,
        bounce: BounceEvent,
        calibration: CalibrationProfile,
    ) -> DecisionResult:
        context = self.fusion.build_context(bounce, calibration)

        explanation: list[str] = []

        if not context.calibration_valid:
            explanation.append("Calibration invalid: automatic call blocked")
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence=0.0,
                context=context,
                explanation=tuple(explanation + list(context.reasons)),
            )

        if context.reasons:
            explanation.extend(context.reasons)
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence=0.0,
                context=context,
                explanation=tuple(explanation),
            )

        if bounce.confidence < self.min_bounce_confidence:
            explanation.append(
                f"Bounce confidence {bounce.confidence:.3f} below "
                f"{self.min_bounce_confidence:.3f}"
            )
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence=bounce.confidence,
                context=context,
                explanation=tuple(explanation),
            )

        d = context.signed_distance_m
        u = context.total_uncertainty_m

        if d is None or u is None:
            explanation.append("Missing geometry evidence")
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence=0.0,
                context=context,
                explanation=tuple(explanation),
            )

        explanation.append(
            f"Nearest line: {context.nearest_line}"
        )
        explanation.append(
            f"Signed distance: {d * 1000.0:.2f} mm"
        )
        explanation.append(
            f"Estimated uncertainty: ±{u * 1000.0:.2f} mm"
        )

        if abs(d) <= u:
            explanation.append(
                "Uncertainty band overlaps court boundary"
            )
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence=min(0.69, bounce.confidence),
                context=context,
                explanation=tuple(explanation),
            )

        decision = Decision.IN if d > 0 else Decision.OUT
        confidence = self._confidence_from_margin(
            d,
            u,
            bounce.confidence,
        )

        explanation.append(
            "Point is safely inside uncertainty band"
            if decision == Decision.IN
            else "Point is safely outside uncertainty band"
        )

        return DecisionResult(
            decision=decision,
            confidence=confidence,
            context=context,
            explanation=tuple(explanation),
        )
