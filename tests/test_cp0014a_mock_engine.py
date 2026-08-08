import numpy as np

from linecaller.proposals.mock_engine import MockProposalEngine
from linecaller.proposals.models import ProposalStatus


def test_mock_engine():
    frame = np.zeros((100,100,3), dtype=np.uint8)

    result = MockProposalEngine().propose(7, frame)

    assert result.frame_number == 7
    assert len(result.proposals) == 1
    assert result.proposals[0].status == ProposalStatus.AUTO
