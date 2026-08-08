import pytest

from linecaller.live.metrics import LiveMetrics


def test_metrics_fps():
    m = LiveMetrics()

    m.record_frame(10.0)
    m.record_frame(10.0)

    assert m.avg_processing_ms == pytest.approx(10.0)
    assert m.processing_fps == pytest.approx(100.0)
