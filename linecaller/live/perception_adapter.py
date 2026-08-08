from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .pipeline_models import LivePipelineResult


class LivePerceptionAdapter(ABC):
    @abstractmethod
    def process(
        self,
        *,
        frame_number: int,
        frame: np.ndarray,
        timestamp: float,
    ) -> LivePipelineResult:
        raise NotImplementedError
