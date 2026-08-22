from __future__ import annotations

from fastapi.testclient import TestClient

from linecaller.api.live_app import app
from linecaller.dcf.external_grid_frame_loop import BallComponent
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
)
from linecaller.dcf.predicted_topk_selector import (
    PredictedTopKSelector,
)


def _c(
    x: float,
    y: float,
    *,
    scale: float = 3.0,
    recovered: bool = False,
) -> BallComponent:
    return BallComponent(
        bbox=(int(round(x)) - 1, int(round(y)) - 1, 3, 3),
        area=7,
        centroid_xy=(float(x), float(y)),
        scale_px=float(scale),
        recovered=bool(recovered),
        recovery_age=1 if recovered else 0,
    )


def test_bootstrap_locks_straight_ball_and_excludes_52px_false_blob():
    selector = PredictedTopKSelector(
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
        min_gate_px=5.0,
        gate_scale=4.75,
        bootstrap_min_straightness=0.70,
    )
    frames = (
        (1, (_c(237.7, 64.0),)),
        (2, (_c(237.1, 65.1),)),
    )
    real = _c(237.2, 65.7)
    false = _c(286.1, 55.7)

    selection = selector.select(3, [real, false], frames)

    assert selection.state == "BOOTSTRAP"
    assert selector.locked
    assert real in selection.selected
    assert false not in selection.selected
    assert selection.predicted_xy is not None
    assert abs(
        selection.predicted_xy[0] - false.centroid_xy[0]
    ) > 40.0


def test_locked_track_does_not_jump_to_far_blob_when_ball_is_missing():
    selector = PredictedTopKSelector(
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
        min_gate_px=5.0,
        gate_scale=4.75,
        max_misses=2,
    )
    frames = (
        (1, (_c(100.0, 100.0),)),
        (2, (_c(101.0, 101.0),)),
    )
    selection = selector.select(
        3,
        [_c(102.0, 102.0)],
        frames,
    )
    assert selection.state == "BOOTSTRAP"

    far = _c(154.0, 90.0)
    missing = selector.select(4, [far], ())
    assert missing.state == "PREDICTED"
    assert missing.selected == ()
    assert far not in missing.selected
    assert selector.locked


def test_locked_track_returns_only_three_nearest_components():
    selector = PredictedTopKSelector(
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
        min_gate_px=5.0,
        gate_scale=4.75,
    )
    frames = (
        (1, (_c(50.0, 50.0),)),
        (2, (_c(52.0, 51.0),)),
    )
    selector.select(
        3,
        [_c(54.0, 52.0)],
        frames,
    )

    current = [
        _c(56.0, 53.0),
        _c(56.8, 53.2),
        _c(55.4, 52.7),
        _c(58.0, 54.0),
        _c(90.0, 20.0),
    ]
    selection = selector.select(4, current, ())

    assert selection.state == "LOCKED"
    assert len(selection.selected) == 3
    assert current[-1] not in selection.selected


def test_recovered_components_cannot_bootstrap_independent_lock():
    selector = PredictedTopKSelector(
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
    )
    frames = (
        (1, (_c(10.0, 10.0, recovered=True),)),
        (2, (_c(11.0, 11.0, recovered=True),)),
    )
    selection = selector.select(
        3,
        [_c(12.0, 12.0, recovered=True)],
        frames,
    )

    assert selection.state == "UNLOCKED"
    assert not selector.locked
    assert selection.selected == ()


def test_historical_default_off_and_wizard_health_top3_on():
    cfg = OfficialExternalLiveConfig()
    assert cfg.predicted_top3_candidates is False
    assert cfg.trajectory_candidate_top_k == 3

    client = TestClient(app)
    health = client.get("/health").json()
    assert (
        health["predicted_top3_feature_version"]
        == "CP-0036.2.4.6"
    )

    html = client.get("/").text
    for token in (
        'id="trackLock"',
        'id="top3Selected"',
        'id="top3Rejects"',
        'id="trackPredXY"',
        'id="top3XY"',
        "TOP3_REJECT",
    ):
        assert token in html


def test_cp0036_2_4_6_source_blocks_are_unique():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    recovery = (
        root / "linecaller/dcf/contact_appearance_recovery.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")
    wizard = (
        root / "linecaller/api/static/wizard.html"
    ).read_text(encoding="utf-8")

    assert recovery.count(
        "predicted_top3_candidates: bool = False"
    ) == 1
    assert recovery.count(
        "def _trajectory_component_for_candidate("
    ) == 1
    assert live.count(
        '"predicted_top3_feature_version": "CP-0036.2.4.6"'
    ) == 1
    assert wizard.count('id="trackLock"') == 1
    assert wizard.count("TOP3_REJECT x${top3Rejects}") == 1
