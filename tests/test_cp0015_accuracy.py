import pytest

from linecaller.live.accuracy import AccuracyGate


def test_accuracy_gate_passes_98_percent():
    gate = AccuracyGate(
        target=.98,
        minimum_samples=100,
    )

    result = gate.evaluate(
        correct=98,
        incorrect=2,
    )

    assert result.accuracy == pytest.approx(.98)
    assert result.passed is True


def test_accuracy_gate_requires_sample_size():
    gate = AccuracyGate(
        target=.98,
        minimum_samples=500,
    )

    result = gate.evaluate(
        correct=99,
        incorrect=1,
    )

    assert result.accuracy == pytest.approx(.99)
    assert result.passed is False
