from __future__ import annotations

import numpy as np

from .confidence import status_from_confidence
from .engine import ProposalEngine
from .models import (
    BallProposal,
    ProposalResult,
    ProposalSource,
)


class MockProposalEngine(ProposalEngine):
    """
    Deterministic proposal engine used for tests and UI integration.
    """

    def __init__(
        self,
        *,
        x: float = 100.0,
        y: float = 120.0,
        width: float = 12.0,
        height: float = 12.0,
        confidence: float = 0.95,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def propose(
        self,
        frame_number: int,
        frame: np.ndarray,
    ) -> ProposalResult:

        proposal = BallProposal(
            frame_number=frame_number,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            confidence=self.confidence,
            source=ProposalSource.MOCK,
            status=status_from_confidence(self.confidence),
        )

        return ProposalResult(
            frame_number=frame_number,
            proposals=(proposal,),
            engine_name="MockProposalEngine",
            latency_ms=0.0,
        )
