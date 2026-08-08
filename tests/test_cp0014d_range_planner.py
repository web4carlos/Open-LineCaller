from linecaller.dataset.models import BallBox
from linecaller.dataset.range_planner import RangeProposalPlanner
from linecaller.dataset.studio_session import DatasetStudioSession


def session():
    return DatasetStudioSession(
        clip_id="x",
        source_video="x.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=100,
    )


def test_planner_skips_confirmed():
    s = session()
    s.set_ball_box(5, BallBox(1,1,5,5))

    frames = RangeProposalPlanner().eligible_frames(
        s,
        start_frame=4,
        end_frame=6,
    )

    assert frames == (4,6)
