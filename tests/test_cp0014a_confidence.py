from linecaller.proposals.confidence import status_from_confidence
from linecaller.proposals.models import ProposalStatus


def test_confidence_statuses():
    assert status_from_confidence(.95) == ProposalStatus.AUTO
    assert status_from_confidence(.75) == ProposalStatus.REVIEW
    assert status_from_confidence(.20) == ProposalStatus.REJECT
