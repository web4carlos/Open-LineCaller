from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .models import ProposalResult


class ProposalEngine(ABC):
    @abstractmethod
    def propose(
        self,
        frame_number: int,
        frame: np.ndarray,
    ) -> ProposalResult:
        raise NotImplementedError
