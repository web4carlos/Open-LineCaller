from __future__ import annotations

from pathlib import Path

from linecaller.dcf.player_net_half_e2e_truth import (
    FEATURE_VERSION,
    _build_runtime,
    run_player_net_half_e2e_truth_gate,
)
from linecaller.dcf.player_net_half_synthetic import (
    PlayerNetHalfSyntheticReference,
    SyntheticHalfCourtConfig,
)


def _result(side: str):
    result = run_player_net_half_e2e_truth_gate(side)
    assert result.passed, "\n".join(result.failures)
    return result


def test_right_rendered_frames_pass_official_e2e_truth():
    result = _result("RIGHT")
    assert result.feature_version == FEATURE_VERSION
    assert result.mount_side == "RIGHT"
    assert result.frame_count == 75
    assert result.pose_safe_frames == 75
    assert not result.scan_suppressed_frames


def test_left_rendered_frames_pass_official_e2e_truth():
    result = _result("LEFT")
    assert result.mount_side == "LEFT"
    assert result.frame_count == 75
    assert result.pose_safe_frames == 75


def test_e2e_truth_has_exact_out_left_and_safe_in():
    for side in ("RIGHT", "LEFT"):
        result = _result(side)
        assert len(result.official_calls) == 1
        call = result.official_calls[0]
        assert call.contact_frame == 24
        assert call.region == "OUT_LEFT"
        assert 25 <= call.confirm_frame <= 26
        assert not result.in_scenario_out_frames
        assert not result.post_in_out_frames


def test_e2e_truth_blocks_deep_distractor_acquisition_and_reacquisition():
    for side in ("RIGHT", "LEFT"):
        result = _result(side)
        assert not result.first_deep_locked_frames
        assert not result.reacquire_deep_locked_frames
        assert any(9 <= frame <= 28 for frame in result.bootstrap_frames)
        assert any(37 <= frame <= 57 for frame in result.bootstrap_frames)
        assert any(
            9 <= frame <= 28
            for frame in result.ingress_accepted_frames
        )
        assert any(
            37 <= frame <= 57
            for frame in result.ingress_accepted_frames
        )


def test_e2e_truth_uses_rendered_detector_and_outside_only_runtime():
    result = _result("RIGHT")
    assert result.frames_with_ball_components > 0
    assert result.raw_z0_total > 0
    assert result.up_total > 0
    assert result.outside_only_runtime is True
    assert result.player_image_down_gate_enabled is False


def test_player_net_ingress_disables_only_oblique_invalid_image_down_gate():
    scene = PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side("RIGHT")
    )
    runtime = _build_runtime(scene, "RIGHT")
    loop = getattr(runtime, "_frame_loop")
    assert loop is not None
    assert loop.trajectory_net_ingress_guard is True
    assert loop.approach_direction_gate is False


def test_health_exposes_cp0036_2_8():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    health = TestClient(app).get("/health").json()
    assert (
        health["player_net_half_e2e_truth_feature_version"]
        == "CP-0036.2.8"
    )


def test_cp0036_2_8_structural_invariants():
    root = Path(__file__).resolve().parents[1]
    module = (
        root / "linecaller/dcf/player_net_half_e2e_truth.py"
    ).read_text(encoding="utf-8")
    runtime = (
        root / "linecaller/dcf/official_external_live_runtime.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")

    assert module.count(
        'FEATURE_VERSION = "CP-0036.2.8"'
    ) == 1
    assert module.count(
        "def run_player_net_half_e2e_truth_gate("
    ) == 1
    assert "OUTSIDE GRID" not in module or "outside" in module.lower()
    assert runtime.count(
        "approach_direction_gate=("
    ) == 1
    assert runtime.count(
        "not c.trajectory_net_ingress_guard"
    ) == 1
    assert live.count(
        '"player_net_half_e2e_truth_feature_version": "CP-0036.2.8"'
    ) == 1
