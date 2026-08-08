from linecaller.proposals.models import (
    BallProposal,
    ProposalSource,
)


def test_proposal_properties():
    p = BallProposal(
        frame_number=1,
        x=10,
        y=20,
        width=8,
        height=6,
        confidence=.9,
        source=ProposalSource.MOCK,
    )

    assert p.center_x == 14
    assert p.center_y == 23
    assert p.area == 48
    assert p.validate() == ()


def test_proposal_normalizes_confidence():
    p = BallProposal(
        frame_number=1,
        x=0,
        y=0,
        width=10,
        height=10,
        confidence=4.0,
    ).normalized()

    assert p.confidence == 1.0
