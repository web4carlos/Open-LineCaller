import numpy as np

from linecaller.proposals.models import ProposalSource
from linecaller.proposals.yolo_engine import YOLOProposalEngine


class FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeArray([
            [10, 20, 30, 40],
            [100, 110, 140, 150],
        ])
        self.conf = FakeArray([0.91, 0.80])
        self.cls = FakeArray([32, 0])


class FakeResult:
    boxes = FakeBoxes()
    names = {
        0: "person",
        32: "sports ball",
    }


class FakeBackend:
    names = FakeResult.names

    def predict(self, **kwargs):
        return [FakeResult()]


def test_yolo_engine_converts_sports_ball():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    engine = YOLOProposalEngine(
        backend=FakeBackend(),
        class_name="sports ball",
    )

    result = engine.propose(7, frame)

    assert len(result.proposals) == 1

    proposal = result.proposals[0]

    assert proposal.frame_number == 7
    assert proposal.x == 10
    assert proposal.y == 20
    assert proposal.width == 20
    assert proposal.height == 20
    assert proposal.confidence == 0.91
    assert proposal.source == ProposalSource.YOLO


def test_yolo_engine_can_disable_class_filter():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    engine = YOLOProposalEngine(
        backend=FakeBackend(),
        class_name=None,
    )

    result = engine.propose(0, frame)

    assert len(result.proposals) == 2


def test_yolo_engine_respects_confidence():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    engine = YOLOProposalEngine(
        backend=FakeBackend(),
        class_name=None,
        min_confidence=.85,
    )

    result = engine.propose(0, frame)

    assert len(result.proposals) == 1
