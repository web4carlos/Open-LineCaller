from linecaller.dataset.range_planner import RangeProposalPlanner
from linecaller.dataset.studio_session import DatasetStudioSession


def test_span_is_limited():
    s = DatasetStudioSession(
        clip_id="x",
        source_video="x.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=200,
    )

    frames = RangeProposalPlanner(
        max_span=10,
    ).eligible_frames(
        s,
        start_frame=0,
        end_frame=100,
    )

    assert frames[0] == 0
    assert frames[-1] == 10
