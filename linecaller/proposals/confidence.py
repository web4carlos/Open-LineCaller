from __future__ import annotations

from .models import ProposalStatus


def status_from_confidence(
    confidence: float,
    *,
    auto_threshold: float = 0.90,
    review_threshold: float = 0.60,
) -> ProposalStatus:
    c = max(0.0, min(1.0, float(confidence)))

    if c >= auto_threshold:
        return ProposalStatus.AUTO
    if c >= review_threshold:
        return ProposalStatus.REVIEW
    return ProposalStatus.REJECT
