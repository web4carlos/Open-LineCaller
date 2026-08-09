from linecaller.product.match_history import MatchHistoryStore
from linecaller.product.summary_models import MatchSummary


def make_summary():
    return MatchSummary(
        match_id="m1",
        started_at=1,
        ended_at=2,
        duration_seconds=1,
        camera_name="Camera 0",
        calibration_mode="AUTO",
        calls_total=1,
        calls_in=1,
        calls_out=0,
        calls_review=0,
        average_fps=60,
        average_latency_ms=40,
        average_confidence=.99,
        tracking_losses=0,
        replay_count=0,
    )


def test_history_roundtrip(tmp_path):
    path = tmp_path / "history.jsonl"
    store = MatchHistoryStore(path)

    store.append(make_summary())

    rows = store.load_all()

    assert len(rows) == 1
    assert rows[0]["match_id"] == "m1"
