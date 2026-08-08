import numpy as np

from linecaller.proposals.yolo_engine import YOLOProposalEngine


class BrokenBackend:
    def predict(self, **kwargs):
        raise RuntimeError("boom")


def test_fail_open_returns_empty_result():
    engine = YOLOProposalEngine(
        backend=BrokenBackend(),
        fail_open=True,
    )

    result = engine.propose(
        1,
        np.zeros((20,20,3), dtype=np.uint8),
    )

    assert result.proposals == ()
    assert engine.last_error is not None
