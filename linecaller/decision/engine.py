from __future__ import annotations

from dataclasses import dataclass

from .fusion import CourtFusion
from .models import (
    Decision,
    DecisionContext,
    DecisionResult as LegacyDecisionResult,
)


@dataclass(frozen=True)
class GeometryDecisionResult:
    call: str
    confidence: float
    reason: str
    geometry_state: str
    nearest_line: str
    signed_distance_in: float


# Public compatibility alias used by linecaller.decision.__init__
DecisionResult = GeometryDecisionResult


class DecisionEngine:
    """
    Backward-compatible CP-0029 Decision Engine.

    Supports BOTH contracts:

    Legacy:
        DecisionEngine(
            fusion=CourtFusion(...),
            min_bounce_confidence=.70,
        )

        result = engine.decide(
            bounce_event,
            calibration_profile,
        )

    CP-0029 geometry API:
        result = engine.decide(
            geometry_state="INSIDE",
            nearest_line="LEFT_SIDELINE",
            signed_distance_ft=0.5,
            bounce_confidence=0.8,
            bounce_score=0.8,
        )

    Legacy decisions are uncertainty-aware through CourtFusion.
    Geometry decisions use the newer conservative line-contact policy.
    """

    def __init__(
        self,
        *,
        fusion: CourtFusion | None = None,
        min_bounce_confidence: float = 0.35,
        min_bounce_score: float = 0.40,
        review_band_in: float = 3.0,
        ball_contact_radius_in: float = 1.45,
    ):
        self.fusion = fusion or CourtFusion()
        self.min_bounce_confidence = float(min_bounce_confidence)
        self.min_bounce_score = float(min_bounce_score)
        self.review_band_in = float(review_band_in)
        self.ball_contact_radius_in = float(ball_contact_radius_in)

    def decide(self, *args, **kwargs):
        # Legacy API: decide(bounce, calibration)
        if len(args) == 2 and not kwargs:
            return self._decide_legacy(
                args[0],
                args[1],
            )

        # CP-0029 API: keyword-only geometry inputs.
        if kwargs:
            return self._decide_geometry(**kwargs)

        raise TypeError(
            "DecisionEngine.decide expects either "
            "(bounce, calibration) or geometry keyword arguments."
        )

    def _decide_legacy(
        self,
        bounce,
        calibration,
    ) -> LegacyDecisionResult:
        context: DecisionContext = self.fusion.build_context(
            bounce,
            calibration,
        )

        explanation = []

        if bounce.confidence < self.min_bounce_confidence:
            explanation.append(
                "Bounce confidence below minimum threshold"
            )
            return LegacyDecisionResult(
                decision=Decision.REVIEW,
                confidence=float(bounce.confidence),
                context=context,
                explanation=tuple(explanation),
            )

        if not context.calibration_valid:
            explanation.append(
                "Calibration is invalid"
            )
            return LegacyDecisionResult(
                decision=Decision.REVIEW,
                confidence=0.0,
                context=context,
                explanation=tuple(explanation),
            )

        if (
            context.signed_distance_m is None
            or context.total_uncertainty_m is None
        ):
            explanation.extend(
                context.reasons
                or ("Geometry context is incomplete",)
            )
            return LegacyDecisionResult(
                decision=Decision.REVIEW,
                confidence=0.0,
                context=context,
                explanation=tuple(explanation),
            )

        distance = float(context.signed_distance_m)
        uncertainty = float(context.total_uncertainty_m)

        # If the uncertainty band crosses the boundary, force REVIEW.
        if abs(distance) <= uncertainty:
            explanation.append(
                "Boundary lies inside uncertainty band"
            )
            return LegacyDecisionResult(
                decision=Decision.REVIEW,
                confidence=max(
                    0.0,
                    min(
                        1.0,
                        bounce.confidence * 0.5,
                    ),
                ),
                context=context,
                explanation=tuple(explanation),
            )

        if distance > 0:
            explanation.append(
                "Bounce is clearly inside court boundary"
            )
            decision = Decision.IN
        else:
            explanation.append(
                "Bounce is clearly outside court boundary"
            )
            decision = Decision.OUT

        # Confidence grows as distance separates from uncertainty.
        separation = max(
            0.0,
            abs(distance) - uncertainty,
        )
        geometric_conf = min(
            1.0,
            separation / max(
                uncertainty,
                1e-9,
            ),
        )

        confidence = max(
            0.0,
            min(
                1.0,
                0.5 * float(bounce.confidence)
                + 0.5 * geometric_conf,
            ),
        )

        return LegacyDecisionResult(
            decision=decision,
            confidence=confidence,
            context=context,
            explanation=tuple(explanation),
        )

    def _decide_geometry(
        self,
        *,
        geometry_state: str,
        nearest_line: str,
        signed_distance_ft: float,
        bounce_confidence: float,
        bounce_score: float,
    ) -> GeometryDecisionResult:
        state = str(geometry_state).upper()
        signed_in = float(signed_distance_ft) * 12.0

        evidence = min(
            max(float(bounce_confidence), 0.0),
            max(float(bounce_score), 0.0),
        )

        if (
            float(bounce_confidence)
            < self.min_bounce_confidence
            or float(bounce_score)
            < self.min_bounce_score
        ):
            return GeometryDecisionResult(
                "REVIEW",
                evidence,
                "LOW_BOUNCE_CONFIDENCE",
                state,
                nearest_line,
                signed_in,
            )

        if signed_in >= self.ball_contact_radius_in:
            return GeometryDecisionResult(
                "IN",
                evidence,
                "CLEARLY_INSIDE",
                state,
                nearest_line,
                signed_in,
            )

        if signed_in <= -self.review_band_in:
            return GeometryDecisionResult(
                "OUT",
                evidence,
                "CLEARLY_OUTSIDE",
                state,
                nearest_line,
                signed_in,
            )

        if (
            -self.ball_contact_radius_in
            <= signed_in
            <= self.ball_contact_radius_in
        ):
            return GeometryDecisionResult(
                "IN",
                evidence,
                "BALL_CONTACTS_LINE",
                state,
                nearest_line,
                signed_in,
            )

        return GeometryDecisionResult(
            "REVIEW",
            evidence,
            "TOO_CLOSE_TO_CALL",
            state,
            nearest_line,
            signed_in,
        )

