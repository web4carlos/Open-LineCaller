from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OfficiatingDecision:
    frame: int
    base_decision: str
    final_decision: str
    force_review: bool
    gate_decision: str
    confidence: float
    reason: str


class ForceReviewEnforcer:
    """
    CP-0029.1

    Final safety layer between geometric decision and officiating output.

    VERIFIED_BOUNCE:
        base IN/OUT/REVIEW is preserved.

    REVIEW_BOUNCE:
        final decision is always REVIEW.

    REJECTED_BOUNCE / unknown:
        final decision is REVIEW and marked blocked.
        Normally these events should already have been dropped upstream.
    """

    def enforce(
        self,
        *,
        frame: int,
        base_decision: str,
        gate_decision: str,
        gate_confidence: float,
        force_review: bool = False,
    ) -> OfficiatingDecision:

        base = str(base_decision).strip().upper()
        gate = str(gate_decision).strip().upper()
        confidence = max(0.0, min(1.0, float(gate_confidence)))

        if gate == "VERIFIED_BOUNCE" and not force_review:
            return OfficiatingDecision(
                int(frame),
                base,
                base,
                False,
                gate,
                confidence,
                "VERIFIED_BOUNCE_BASE_DECISION_PRESERVED",
            )

        if gate == "REVIEW_BOUNCE" or force_review:
            return OfficiatingDecision(
                int(frame),
                base,
                "REVIEW",
                True,
                gate,
                confidence,
                "BOUNCE_EVIDENCE_FORCES_REVIEW",
            )

        return OfficiatingDecision(
            int(frame),
            base,
            "REVIEW",
            True,
            gate,
            0.0,
            "UNSAFE_OR_UNKNOWN_GATE_STATE",
        )
