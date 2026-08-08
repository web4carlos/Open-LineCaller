from linecaller.live.models import LiveFramePacket
from linecaller.live.replay import ReplayBuffer


def test_replay_snapshot():
    buffer = ReplayBuffer(max_frames=20)

    for frame in range(10):
        buffer.append(
            LiveFramePacket(
                frame_number=frame,
                captured_at=float(frame),
                frame=f"frame-{frame}",
            )
        )

    replay = buffer.snapshot(
        event_frame=7,
        pre_frames=2,
        post_frames=1,
    )

    assert [p.frame_number for p in replay] == [5,6,7,8]
