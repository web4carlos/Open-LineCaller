from linecaller.apps.live_controller import LiveMatchController
from linecaller.apps.live_state import MatchRunState
from linecaller.live.models import LiveDecision, LiveEvidence


def evidence(decision):
    return LiveEvidence(
        frame_number=10,
        decision=decision,
        confidence=.95,
        replay_requested=decision == LiveDecision.REVIEW,
        decision_latency_ms=20,
        processing_fps=60,
    )


def test_controller_lifecycle():
    c = LiveMatchController()

    c.set_ready(calibration_valid=True)
    assert c.state.calibration_status == "VALID"

    c.start()
    assert c.state.run_state == MatchRunState.LIVE

    c.stop()
    assert c.state.run_state == MatchRunState.STOPPED


def test_review_activates_replay():
    c = LiveMatchController()
    c.handle_evidence(evidence(LiveDecision.REVIEW))

    assert c.state.last_call == "REVIEW"
    assert c.state.replay_active is True


def test_out_is_normal_call():
    c = LiveMatchController()
    c.handle_evidence(evidence(LiveDecision.OUT))

    assert c.state.last_call == "OUT"
    assert c.state.replay_active is False
