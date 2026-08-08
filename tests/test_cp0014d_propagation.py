from linecaller.dataset.models import BallBox
from linecaller.dataset.propagation import AnnotationPropagator
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.proposals.models import BallProposal


def proposal(frame,x):
    return BallProposal(
        frame_number=frame,
        x=x,
        y=10,
        width=5,
        height=5,
        confidence=.9,
    )


def test_propagation_never_overwrites_confirmed():
    s = DatasetStudioSession(
        clip_id="x",
        source_video="x.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=10,
    )

    s.set_ball_box(2, BallBox(50,50,8,8))

    count = AnnotationPropagator().propagate(
        session=s,
        predictions=[
            (1, proposal(1,10)),
            (2, proposal(2,20)),
            (3, proposal(3,30)),
        ],
    )

    assert count == 2
    assert s.frame_annotation(2).ball.x == 50
