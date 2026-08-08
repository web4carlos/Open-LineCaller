from __future__ import annotations

from dataclasses import dataclass

from .geometry import signed_distances_to_court
from .models import BounceEvent, Call, CourtModel, DecisionResult


@dataclass
class DecisionEngine:
    review_margin_m: float = 0.015
    min_bounce_confidence: float = 0.80

    def evaluate(self, bounce: BounceEvent, court: CourtModel) -> DecisionResult:
        if bounce.confidence < self.min_bounce_confidence:
            return DecisionResult(
                call=Call.REVIEW,
                confidence=bounce.confidence,
                signed_distance_m=0.0,
                reason='Bounce confidence below threshold',
            )

        distances = signed_distances_to_court(bounce.x_m, bounce.y_m, court)
        d = distances.minimum_signed

        # Court boundary lines are part of the court. A point slightly outside can
        # still be REVIEW if it falls inside the configured uncertainty margin.
        if d >= 0:
            call = Call.IN
            reason = 'Bounce point is inside or on the court boundary'
        elif abs(d) <= self.review_margin_m:
            call = Call.REVIEW
            reason = 'Bounce is too close to the boundary for a reliable call'
        else:
            call = Call.OUT
            reason = 'Bounce point is outside the court boundary'

        confidence = max(0.0, min(1.0, bounce.confidence))

        return DecisionResult(
            call=call,
            confidence=confidence,
            signed_distance_m=d,
            reason=reason,
        )
