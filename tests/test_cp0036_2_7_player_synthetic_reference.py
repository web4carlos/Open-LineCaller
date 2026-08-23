from __future__ import annotations

import json
from pathlib import Path

from linecaller.dcf.player_net_half_synthetic import (
    FEATURE_VERSION,
    PlayerNetHalfSyntheticReference,
    run_net_ingress_selector_gate,
)


def test_synthetic_reference_is_player_half_court():
    scene = PlayerNetHalfSyntheticReference()
    assert scene.calibration.coverage == "HALF_COURT"
    assert scene.calibration.config.half_court_y_bu == 91.0
    assert scene.calibration.image_size == (960, 540)


def test_synthetic_reference_has_known_out_and_in_contacts():
    scene = PlayerNetHalfSyntheticReference()
    contacts = [f for f in scene.truth_frames() if f.contact]
    assert [(f.frame_no, f.expected_call) for f in contacts] == [
        (24, "OUT_LEFT"),
        (52, "IN"),
    ]
    out_contact, in_contact = contacts
    assert out_contact.ball_floor_xy_bu is not None
    assert out_contact.ball_floor_xy_bu[0] < 0.0
    assert in_contact.ball_floor_xy_bu is not None
    x_bu, y_bu = in_contact.ball_floor_xy_bu
    assert 0.0 < x_bu < 83.0
    assert 0.0 < y_bu < 91.0


def test_real_synthetic_ball_starts_at_net_and_moves_inward():
    scene = PlayerNetHalfSyntheticReference()
    rally = [
        f
        for f in scene.truth_frames()
        if f.scenario == "INGRESS_OUT_LEFT"
        and f.ball_floor_xy_bu is not None
    ]
    start_y = rally[0].ball_floor_xy_bu[1]
    contact_y = next(f for f in rally if f.contact).ball_floor_xy_bu[1]
    assert start_y >= 89.0
    assert contact_y < 25.0
    assert start_y - contact_y > 60.0


def test_deep_distractor_stays_outside_ingress_band():
    scene = PlayerNetHalfSyntheticReference()
    points = [
        f.distractor_floor_xy_bu
        for f in scene.truth_frames()
        if f.scenario == "DEEP_DISTRACTOR_ONLY"
    ]
    assert points
    assert all(p is not None and p[1] < 55.0 for p in points)


def test_synthetic_net_ingress_selector_gate():
    result = run_net_ingress_selector_gate()
    assert result["feature_version"] == FEATURE_VERSION
    assert result["passed"] is True
    assert result["deep_state"] == "UNLOCKED"
    assert result["deep_ingress_rejections"] > 0
    assert result["real_state"] == "BOOTSTRAP"
    assert result["real_reason"] == "ACCEPTED"
    assert result["real_inward_bu"] > 1.0
    assert result["lock_error_px"] < 1e-6


def test_reference_images_have_expected_shapes():
    scene = PlayerNetHalfSyntheticReference()
    assert scene.background_rgb().shape == (540, 960, 3)
    assert scene.ball_template_rgb().shape == (48, 48, 3)


def test_bundle_writes_reproducible_core_assets(tmp_path):
    scene = PlayerNetHalfSyntheticReference()
    bundle = scene.build_bundle(
        tmp_path / "reference",
        write_video=False,
        write_frames=False,
    )
    for raw in (
        bundle.background,
        bundle.calibration,
        bundle.ball_template,
        bundle.truth,
    ):
        assert Path(raw).exists()

    truth = json.loads(Path(bundle.truth).read_text(encoding="utf-8"))
    assert truth["feature_version"] == FEATURE_VERSION
    assert truth["product_mode"] == "PLAYER NET HALF"
    assert truth["coverage"] == "HALF_COURT"
    assert truth["coordinate_convention"]["outside_grid_only"] is True
    assert len(truth["frames"]) == bundle.frame_count == 75


def test_health_exposes_cp0036_2_7():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    health = TestClient(app).get("/health").json()
    assert (
        health["player_synthetic_reference_feature_version"]
        == "CP-0036.2.7"
    )


def test_cp0036_2_7_structural_uniqueness():
    root = Path(__file__).resolve().parents[1]
    module = (
        root / "linecaller/dcf/player_net_half_synthetic.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")
    assert module.count("class PlayerNetHalfSyntheticReference:") == 1
    assert module.count("def run_net_ingress_selector_gate()") == 1
    assert live.count(
        '"player_synthetic_reference_feature_version": "CP-0036.2.7"'
    ) == 1
