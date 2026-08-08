from linecaller.proposals.dataset_adapter import proposal_to_ball_box
from linecaller.proposals.models import BallProposal


def test_dataset_adapter():
    proposal = BallProposal(
        frame_number=1,
        x=10,
        y=20,
        width=8,
        height=6,
        confidence=.9,
    )

    box = proposal_to_ball_box(proposal)

    assert box.x == 10
    assert box.height == 6
