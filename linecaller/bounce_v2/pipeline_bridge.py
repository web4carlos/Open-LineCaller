from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class BridgeAction(str, Enum):
    AUTO_OFFICIATE = "AUTO_OFFICIATE"
    GEOMETRY_REVIEW_ONLY = "GEOMETRY_REVIEW_ONLY"
    DROP_EVENT = "DROP_EVENT"


@dataclass(frozen=True)
class BridgeResult:
    frame: int
    gate_decision: str
    action: BridgeAction
    force_review: bool
    confidence: float
    reason: str


class OfficiatingPipelineBridge:
    """
    CP-0026.6

    Converts bounce-gate decisions into downstream pipeline permissions.

    VERIFIED_BOUNCE -> may proceed to normal geometry/decision processing.
    REVIEW_BOUNCE   -> may proceed to geometry, but final decision is forced REVIEW.
    REJECTED_BOUNCE -> must not enter officiating geometry.
    """

    def route(self, *, frame: int, gate_decision: str, confidence: float) -> BridgeResult:
        decision = str(gate_decision).strip().upper()
        conf = max(0.0, min(1.0, float(confidence)))

        if decision == "VERIFIED_BOUNCE":
            return BridgeResult(
                int(frame), decision, BridgeAction.AUTO_OFFICIATE,
                False, conf, "VERIFIED_BOUNCE_ALLOWED"
            )

        if decision == "REVIEW_BOUNCE":
            return BridgeResult(
                int(frame), decision, BridgeAction.GEOMETRY_REVIEW_ONLY,
                True, conf, "BOUNCE_REQUIRES_HUMAN_REVIEW"
            )

        return BridgeResult(
            int(frame), decision, BridgeAction.DROP_EVENT,
            False, 0.0, "BOUNCE_REJECTED_BEFORE_GEOMETRY"
        )
