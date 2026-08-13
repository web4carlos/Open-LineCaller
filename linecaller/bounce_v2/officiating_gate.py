from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class BounceGateDecision(str, Enum):
    VERIFIED_BOUNCE = "VERIFIED_BOUNCE"
    REVIEW_BOUNCE = "REVIEW_BOUNCE"
    REJECTED_BOUNCE = "REJECTED_BOUNCE"


@dataclass(frozen=True)
class BounceGateResult:
    decision: BounceGateDecision
    frame: int
    confidence: float
    reason: str


class BounceOfficiatingGate:
    """
    CP-0026.5

    Combines:
      CP-0026.3 physical-contact validation
      CP-0026.4 visual-evidence integrity

    Conservative policy:
      - Physical failure => REJECTED
      - Unverified evidence => REJECTED
      - Observed contacts may be VERIFIED when physical + visual evidence clear
      - Inferred contacts require stronger evidence; otherwise REVIEW
    """

    def __init__(
        self,
        *,
        observed_verify_confidence: float = 0.20,
        inferred_verify_confidence: float = 0.35,
        minimum_review_confidence: float = 0.08,
        minimum_physical_score: float = 0.45,
    ):
        self.observed_verify_confidence = float(observed_verify_confidence)
        self.inferred_verify_confidence = float(inferred_verify_confidence)
        self.minimum_review_confidence = float(minimum_review_confidence)
        self.minimum_physical_score = float(minimum_physical_score)

    def decide(
        self,
        *,
        frame: int,
        physical_valid: bool,
        physical_score: float,
        evidence_accepted: bool,
        evidence_class: str,
        evidence_confidence: float,
    ) -> BounceGateResult:

        if not physical_valid or float(physical_score) < self.minimum_physical_score:
            return BounceGateResult(
                BounceGateDecision.REJECTED_BOUNCE,
                int(frame),
                0.0,
                "PHYSICAL_CONTACT_NOT_VALIDATED",
            )

        if not evidence_accepted or evidence_class == "UNVERIFIED_CONTACT":
            return BounceGateResult(
                BounceGateDecision.REJECTED_BOUNCE,
                int(frame),
                0.0,
                "VISUAL_EVIDENCE_UNVERIFIED",
            )

        p = max(0.0, min(1.0, float(physical_score)))
        e = max(0.0, min(1.0, float(evidence_confidence)))
        combined = 0.60 * p + 0.40 * e

        if evidence_class == "OBSERVED_CONTACT":
            if e >= self.observed_verify_confidence:
                return BounceGateResult(
                    BounceGateDecision.VERIFIED_BOUNCE,
                    int(frame),
                    combined,
                    "OBSERVED_PHYSICAL_CONTACT_VERIFIED",
                )
            if e >= self.minimum_review_confidence:
                return BounceGateResult(
                    BounceGateDecision.REVIEW_BOUNCE,
                    int(frame),
                    combined,
                    "OBSERVED_CONTACT_LOW_VISUAL_CONFIDENCE",
                )

        if evidence_class == "INFERRED_CONTACT":
            if e >= self.inferred_verify_confidence:
                return BounceGateResult(
                    BounceGateDecision.VERIFIED_BOUNCE,
                    int(frame),
                    combined,
                    "INFERRED_CONTACT_STRONGLY_SUPPORTED",
                )
            if e >= self.minimum_review_confidence:
                return BounceGateResult(
                    BounceGateDecision.REVIEW_BOUNCE,
                    int(frame),
                    combined,
                    "INFERRED_CONTACT_REQUIRES_REVIEW",
                )

        return BounceGateResult(
            BounceGateDecision.REJECTED_BOUNCE,
            int(frame),
            combined,
            "INSUFFICIENT_OFFICIATING_EVIDENCE",
        )
