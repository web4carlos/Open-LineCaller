import numpy as np

from linecaller.dataset.assisted_controller import AssistedAnnotationController
from linecaller.dataset.models import BallBox
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.proposals.mock_engine import MockProposalEngine


def session():
    return DatasetStudioSession(
        clip_id="x",
        source_video="x.mp4",
        width=640,
        height=480,
        fps=60,
        frame_count=100,
    )


def test_request_and_accept():
    s = session()
    c = AssistedAnnotationController(
        s,
        engine=MockProposalEngine(
            x=10,
            y=20,
            width=8,
            height=8,
            confidence=.95,
        ),
    )

    frame = np.zeros((100,100,3), dtype=np.uint8)

    p = c.request_proposal(5, frame)
    assert p is not None

    box = c.accept_current()
    assert box.width == 8
    assert s.frame_annotation(5).ball is not None
    assert c.metrics.accepted == 1


def test_confirmed_annotation_is_not_overwritten():
    s = session()
    s.set_ball_box(5, BallBox(1,2,3,4))

    c = AssistedAnnotationController(
        s,
        engine=MockProposalEngine(),
    )

    frame = np.zeros((100,100,3), dtype=np.uint8)

    assert c.request_proposal(5, frame) is None
    assert s.frame_annotation(5).ball.x == 1


def test_adjust_counts_adjustment():
    s = session()
    c = AssistedAnnotationController(
        s,
        engine=MockProposalEngine(),
    )

    frame = np.zeros((100,100,3), dtype=np.uint8)
    c.request_proposal(5, frame)

    c.adjust_current(
        5,
        BallBox(20,30,9,9),
    )

    assert c.metrics.adjusted == 1
    assert s.frame_annotation(5).ball.x == 20


def test_reject_counts_rejection():
    s = session()
    c = AssistedAnnotationController(
        s,
        engine=MockProposalEngine(),
    )

    frame = np.zeros((100,100,3), dtype=np.uint8)
    c.request_proposal(5, frame)

    assert c.reject_current() is True
    assert c.metrics.rejected == 1
