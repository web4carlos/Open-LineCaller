from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyGateResult:
    accuracy: float
    correct: int
    incorrect: int
    sample_count: int
    target: float
    minimum_samples: int
    passed: bool


class AccuracyGate:
    def __init__(
        self,
        *,
        target: float = 0.98,
        minimum_samples: int = 500,
    ):
        if not (0.0 < target <= 1.0):
            raise ValueError("target must be within (0,1]")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be > 0")

        self.target = float(target)
        self.minimum_samples = int(minimum_samples)

    def evaluate(
        self,
        *,
        correct: int,
        incorrect: int,
    ) -> AccuracyGateResult:
        correct = int(correct)
        incorrect = int(incorrect)
        total = correct + incorrect

        accuracy = correct / total if total else 0.0

        passed = (
            total >= self.minimum_samples
            and accuracy >= self.target
        )

        return AccuracyGateResult(
            accuracy=accuracy,
            correct=correct,
            incorrect=incorrect,
            sample_count=total,
            target=self.target,
            minimum_samples=self.minimum_samples,
            passed=passed,
        )
