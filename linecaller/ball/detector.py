from abc import ABC, abstractmethod
import numpy as np
from .models import BallCandidate

class BallDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[BallCandidate]:
        raise NotImplementedError
