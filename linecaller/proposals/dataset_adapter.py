from __future__ import annotations

from linecaller.dataset.models import BallBox

from .models import BallProposal


def proposal_to_ball_box(proposal: BallProposal) -> BallBox:
    errors = proposal.validate()
    if errors:
        raise ValueError("; ".join(errors))

    return BallBox(
        x=proposal.x,
        y=proposal.y,
        width=proposal.width,
        height=proposal.height,
    )
