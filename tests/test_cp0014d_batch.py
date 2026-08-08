import numpy as np

from linecaller.dataset.assisted_controller import AssistedAnnotationController
from linecaller.dataset.batch_proposals import BatchProposalRunner
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.proposals.mock_engine import MockProposalEngine


def test_batch_runner():
    s = DatasetStudioSession(
        clip_id="x",
        source_video="x.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=10,
    )

    c = AssistedAnnotationController(
        s,
        engine=MockProposalEngine(),
    )

    runner = BatchProposalRunner()

    items = runner.run(
        session=s,
        controller=c,
        frame_provider=lambda f: np.zeros((20,20,3), dtype=np.uint8),
        start_frame=1,
        end_frame=3,
    )

    assert len(items) == 3
    assert all(item.proposal is not None for item in items)
