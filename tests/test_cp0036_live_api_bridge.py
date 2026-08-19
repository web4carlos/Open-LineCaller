from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app, registry
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridConfig,
)


def _reset_registry():
    registry.runtime = None
    registry._frame_no = 0
    registry.summary.configured = False
    registry.summary.camera_id = None
    registry.summary.mount_position = None
    registry.summary.zones = ()
    registry.summary.receiving_side = None
    registry.summary.calibration_path = None
    registry.summary.background_path = None
    registry.summary.ball_template_path = None
    registry.summary.image_size = None
    registry.summary.active_external_cells = 0
    registry.summary.pose_epoch = 0


def _background():
    w, h = 320, 240
    court = np.array(
        [[60, 210], [260, 210], [220, 50], [100, 50]],
        dtype=np.int32,
    )
    im = np.full((h, w, 3), 28, dtype=np.uint8)
    cv2.polylines(
        im,
        [court.reshape(-1, 1, 2)],
        True,
        (225, 225, 225),
        3,
        cv2.LINE_AA,
    )
    anchors = [
        (60,210),(80,210),(110,210),(145,210),(180,210),
        (215,210),(245,210),(260,210),
        (100,50),(125,50),(155,50),(185,50),(220,50),
        (67,185),(74,160),(82,135),(89,110),(95,80),
        (253,185),(246,160),(239,135),(232,110),(225,80),
    ]
    for i, (x, y) in enumerate(anchors):
        r = 3 + (i % 2)
        cv2.rectangle(im, (x-r,y-r), (x+r,y+r), (245,245,245), -1)
        cv2.line(im, (x-r-2,y), (x+r+2,y), (70,70,70), 1)
        cv2.line(im, (x,y-r-2), (x,y+r+2), (70,70,70), 1)
    return im


def _write_fixture(tmp_path: Path):
    bg = _background()
    bg_path = tmp_path / "background.png"
    Image.fromarray(bg, "RGB").save(bg_path)

    # Tight yellow template with simple texture to avoid a flat/empty profile.
    ball = np.zeros((24, 24, 3), dtype=np.uint8)
    cv2.circle(ball, (12, 12), 9, (250,245,20), -1, cv2.LINE_AA)
    cv2.circle(ball, (9, 9), 2, (255,255,80), -1, cv2.LINE_AA)
    ball_path = tmp_path / "ball.png"
    Image.fromarray(ball, "RGB").save(ball_path)

    cell = ExternalGridCell(
        cell_id=1,
        ix=83,
        iy=30,
        region="OUT_RIGHT",
        top_view_rect_bu=(83.0, 30.0, 84.0, 31.0),
        polygon_image=((244.0,140.0),(258.0,140.0),(258.0,154.0),(244.0,154.0)),
        bbox_image=(243,139,260,156),
        expected_floor_ball_diameter_px=10.0,
    )
    cal = ExternalGridCalibration(
        version=2,
        coverage="FULL_COURT",
        image_size=(320,240),
        image_points=((60.0,210.0),(260.0,210.0),(220.0,50.0),(100.0,50.0)),
        config=ExternalGridConfig(
            court_x_bu=83.0,
            full_court_y_bu=182.0,
            half_court_y_bu=91.0,
            cell_bu=1.0,
            margin_bu=10.0,
        ),
        background_image=str(bg_path),
        image_up_unit=(0.0,-1.0),
        cells=(cell,),
    )
    cal_path = tmp_path / "calibration.json"
    cal.save(cal_path)
    return cal_path, bg_path, ball_path, bg


def test_health_works_without_configuration():
    _reset_registry()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["configured"] is False


def test_status_is_fail_closed_before_configuration():
    _reset_registry()
    client = TestClient(app)
    r = client.get("/api/live/status")
    assert r.status_code == 200
    assert r.json()["configured"] is False

    jpeg = cv2.imencode(".jpg", np.zeros((20,20,3), dtype=np.uint8))[1].tobytes()
    blocked = client.post("/api/live/frame", content=jpeg)
    assert blocked.status_code == 409


def test_configure_paths_creates_official_live_runtime(tmp_path):
    _reset_registry()
    cal, bg, ball, _ = _write_fixture(tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/live/configure-paths",
        json={
            "calibration_path": str(cal),
            "background_path": str(bg),
            "ball_template_path": str(ball),
            "camera_id": "net-right-01",
            "mount_position": "NET_RIGHT",
            "zones": ["FAR_RIGHT"],
            "receiving_side": "FAR",
            "depth_bu": 4,
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["configured"] is True
    assert d["camera_id"] == "net-right-01"
    assert d["receiving_side"] == "FAR"
    assert d["active_external_cells"] == 1
    assert d["image_size"] == [320, 240]


def test_background_frame_reaches_safe_external_runtime(tmp_path):
    _reset_registry()
    cal, bg_path, ball, bg = _write_fixture(tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/live/configure-paths",
        json={
            "calibration_path": str(cal),
            "background_path": str(bg_path),
            "ball_template_path": str(ball),
            "camera_id": "net-right-01",
            "mount_position": "NET_RIGHT",
            "zones": ["FAR_RIGHT"],
            "receiving_side": "FAR",
            "depth_bu": 4,
        },
    )
    assert r.status_code == 200, r.text

    # FastAPI endpoint expects encoded BGR/JPEG; synthetic source is RGB but
    # neutral/gray anchor image, so channel order is not material here.
    ok, encoded = cv2.imencode(".jpg", bg[:, :, ::-1])
    assert ok
    frame = client.post(
        "/api/live/frame?frame_no=1&now_s=1.0&include_image=false",
        content=encoded.tobytes(),
        headers={"Content-Type": "image/jpeg"},
    )
    assert frame.status_code == 200, frame.text
    d = frame.json()
    assert d["pose_state"] == "SAFE"
    assert d["scan_suppressed"] is False
    assert d["active_external_cells"] == 1
    assert d["has_out"] is False


def test_receiving_side_change_can_fail_closed_to_zero_owned_cells(tmp_path):
    _reset_registry()
    cal, bg, ball, _ = _write_fixture(tmp_path)
    client = TestClient(app)

    client.post(
        "/api/live/configure-paths",
        json={
            "calibration_path": str(cal),
            "background_path": str(bg),
            "ball_template_path": str(ball),
            "camera_id": "far-only",
            "mount_position": "NET_RIGHT",
            "zones": ["FAR_RIGHT"],
            "receiving_side": "FAR",
            "depth_bu": 4,
        },
    )
    r = client.post(
        "/api/live/receiving-side",
        json={"side":"NEAR"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["receiving_side"] == "NEAR"
    assert d["active_external_cells"] == 0


def test_root_serves_browser_camera_console():
    _reset_registry()
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "getUserMedia" in r.text
    assert "Open LineCaller" in r.text