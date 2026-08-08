from linecaller.live.engine import LiveOfficiatingEngine
from linecaller.live.models import (
    LiveDecision,
    LiveEvent,
    LiveFramePacket,
)


def engine_with_frames():
    e = LiveOfficiatingEngine(
        replay_pre_frames=2,
        replay_post_frames=1,
    )

    for frame in range(10):
        e.ingest_frame(
            LiveFramePacket(
                frame_number=frame,
                captured_at=float(frame),
                frame=f"f{frame}",
            ),
            processing_ms=10,
        )

    return e


def test_review_requests_replay():
    e = engine_with_frames()

    evidence = e.handle_event(
        LiveEvent(
            frame_number=7,
            decision=LiveDecision.REVIEW,
            confidence=.7,
            event_timestamp=100.0,
        ),
        now=100.05,
    )

    assert evidence.replay_requested is True
    assert evidence.decision == LiveDecision.REVIEW
    assert [p.frame_number for p in e.last_replay] == [5,6,7,8]


def test_out_does_not_request_replay():
    e = engine_with_frames()

    evidence = e.handle_event(
        LiveEvent(
            frame_number=7,
            decision=LiveDecision.OUT,
            confidence=.99,
            event_timestamp=10.0,
        ),
        now=10.02,
    )

    assert evidence.replay_requested is False
    assert evidence.decision == LiveDecision.OUT


def test_duplicate_call_is_suppressed():
    e = LiveOfficiatingEngine()

    first = LiveEvent(
        frame_number=100,
        decision=LiveDecision.OUT,
        confidence=.99,
        event_timestamp=1.0,
    )

    second = LiveEvent(
        frame_number=105,
        decision=LiveDecision.OUT,
        confidence=.99,
        event_timestamp=1.1,
    )

    assert e.handle_event(first, now=1.01).duplicate_suppressed is False
    assert e.handle_event(second, now=1.11).duplicate_suppressed is True
