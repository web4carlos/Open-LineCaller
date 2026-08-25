from __future__ import annotations

import json
import math
from pathlib import Path

from linecaller.dcf.player_net_half_synthetic import (
    CAMERA_GEOMETRY_FEATURE_VERSION,
    LEFT_NET_POST_IMAGE_POINTS,
    RIGHT_NET_POST_IMAGE_POINTS,
    PlayerNetHalfSyntheticReference,
    SyntheticHalfCourtConfig,
    run_net_ingress_selector_gate_for_mount,
)


def test_default_geometry_is_low_right_net_post_oblique():
    cfg = SyntheticHalfCourtConfig()
    assert cfg.camera_mount_side == "RIGHT"
    assert cfg.image_points == RIGHT_NET_POST_IMAGE_POINTS

    p0, p1, p2, p3 = cfg.image_points
    near_width = math.dist(p0, p1)
    far_width = math.dist(p3, p2)

    assert near_width > 2.5 * far_width
    assert p1[1] - p0[1] > 150.0
    assert p1[0] - p2[0] > 250.0
    assert p1[1] - p2[1] > 350.0


def test_left_mount_is_exact_mirror_of_right_mount():
    right = SyntheticHalfCourtConfig.for_mount_side("RIGHT")
    left = SyntheticHalfCourtConfig.for_mount_side("LEFT")
    width = float(right.image_size[0])

    assert left.camera_mount_side == "LEFT"
    assert left.image_points == LEFT_NET_POST_IMAGE_POINTS

    rp = right.image_points
    expected = (
        (width - rp[1][0], rp[1][1]),
        (width - rp[0][0], rp[0][1]),
        (width - rp[3][0], rp[3][1]),
        (width - rp[2][0], rp[2][1]),
    )
    assert left.image_points == expected


def test_oblique_geometry_keeps_player_truth_visible():
    scene = PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side("RIGHT")
    )
    width, height = scene.config.image_size
    projected = [
        scene.project_floor_xy(frame.ball_floor_xy_bu)
        for frame in scene.truth_frames()
        if frame.ball_floor_xy_bu is not None
    ]
    assert projected
    assert all(
        -40.0 <= x <= width + 40.0
        and -40.0 <= y <= height + 40.0
        for x, y in projected
    )


def test_net_ingress_selector_gate_passes_both_mount_sides():
    for side in ("RIGHT", "LEFT"):
        result = run_net_ingress_selector_gate_for_mount(side)
        assert result["passed"] is True
        assert result["real_state"] == "BOOTSTRAP"
        assert result["real_reason"] == "ACCEPTED"
        assert result["lock_error_px"] < 1e-6


def test_manifest_declares_physical_camera_geometry():
    scene = PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side("RIGHT")
    )
    manifest = scene.manifest()

    assert (
        manifest["camera_geometry_feature_version"]
        == CAMERA_GEOMETRY_FEATURE_VERSION
    )
    assert manifest["camera_mount_side"] == "RIGHT"
    assert manifest["camera_model"] == "LOW_NET_POST_OBLIQUE"
    assert manifest["image_points"] == [
        list(p) for p in RIGHT_NET_POST_IMAGE_POINTS
    ]


def test_bundle_preserves_corrected_camera_geometry(tmp_path):
    scene = PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side("LEFT")
    )
    bundle = scene.build_bundle(
        tmp_path / "left",
        write_video=False,
        write_frames=False,
    )
    truth = json.loads(Path(bundle.truth).read_text(encoding="utf-8"))

    assert truth["camera_mount_side"] == "LEFT"
    assert truth["camera_model"] == "LOW_NET_POST_OBLIQUE"
    assert truth["camera_geometry_feature_version"] == "CP-0036.2.7.1"


def test_health_exposes_cp0036_2_7_1():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    health = TestClient(app).get("/health").json()
    assert (
        health["player_net_mount_camera_geometry_feature_version"]
        == "CP-0036.2.7.1"
    )


def test_cp0036_2_7_1_structural_uniqueness():
    root = Path(__file__).resolve().parents[1]
    module = (
        root / "linecaller/dcf/player_net_half_synthetic.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")

    assert module.count(
        'CAMERA_GEOMETRY_FEATURE_VERSION = "CP-0036.2.7.1"'
    ) == 1
    assert module.count("RIGHT_NET_POST_IMAGE_POINTS: tuple") == 1
    assert module.count("LEFT_NET_POST_IMAGE_POINTS: tuple") == 1
    assert live.count(
        '"player_net_mount_camera_geometry_feature_version"'
    ) == 1
