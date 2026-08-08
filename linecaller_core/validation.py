from __future__ import annotations

from dataclasses import dataclass

from .models import Call


@dataclass
class ValidationStats:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    review: int = 0

    @property
    def accuracy(self) -> float:
        judged = self.correct + self.wrong
        if judged == 0:
            return 0.0
        return self.correct / judged


def score_prediction(expected: Call, predicted: Call, stats: ValidationStats) -> None:
    stats.total += 1
    if predicted == Call.REVIEW:
        stats.review += 1
    elif predicted == expected:
        stats.correct += 1
    else:
        stats.wrong += 1
