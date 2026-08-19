from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app, registry
from linecaller.api.live_wizard import wizard_state
from linecaller.dcf.external_grid_frame_loop import ExternalGridCalibration


def _reset():
    wizard_state.reset()
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


def _court_rgb():
    w, h = 320, 240
    im = np.full((h, w, 3), 28, dtype=np.uint8)
    court = np.array(
        [[60,210],[260,210],[220,50],[100,50]],
        dtype=np.int32,
    )
    cv2.polylines(im, [court.reshape(-1,1,2)], True, (230,230,230), 3)
    anchors = [
        (60,210),(80,210),(110,210),(145,210),(180,210),
        (215,210),(245,210),(260,210),(100,50),(125,50),
        (155,50),(185,50),(220,50),(67,185),(74,160),
        (82,135),(89,110),(95,80),(253,185),(246,160),
        (239,135),(232,110),(225,80),
    ]
    for x,y in anchors:
        cv2.rectangle(im,(x-3,y-3),(x+3,y+3),(245,245,245),-1)
    return im


def _png_bytes(rgb):
    buf = BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_root_is_guided_wizard_and_engineering_console_is_preserved():
    _reset()
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "Live Calibration Wizard" in root.text
    assert "Capture Clean Court" in root.text
    assert "Change Ball" in root.text
    assert "getUserMedia" in root.text

    engineering = client.get("/engineering")
    assert engineering.status_code == 200
    assert "Configure live court" in engineering.text


def test_wizard_calibration_builds_only_official_external_cells():
    _reset()
    client = TestClient(app)
    bg = _court_rgb()

    response = client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png_bytes(bg), "image/png")},
        data={
            "image_points": "[[60,210],[260,210],[220,50],[100,50]]",
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ok"] is True
    assert data["external_cells"] > 0
    assert data["next"] == "SELECT_BALL"

    cal = ExternalGridCalibration.load(wizard_state.calibration_path)
    court_x = cal.config.court_x_bu
    court_y = cal.config.court_y_bu(
        __import__(
            "linecaller.dcf.external_grid_frame_loop",
            fromlist=["CalibrationCoverage"],
        ).CalibrationCoverage.parse(cal.coverage)
    )
    for cell in cal.cells:
        x0,y0,x1,y1 = cell.top_view_rect_bu
        assert not (
            x0 >= 0.0 and x1 <= court_x
            and y0 >= 0.0 and y1 <= court_y
        )


def test_ball_click_finishes_runtime_without_manual_files():
    _reset()
    client = TestClient(app)
    bg = _court_rgb()

    cal = client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png_bytes(bg), "image/png")},
        data={
            "image_points": "[[60,210],[260,210],[220,50],[100,50]]",
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert cal.status_code == 200

    ball_frame = bg.copy()
    cv2.circle(ball_frame, (250,145), 6, (250,245,20), -1)

    selected = client.post(
        "/api/wizard/select-ball",
        files={"frame": ("ball.png", _png_bytes(ball_frame), "image/png")},
        data={
            "x": "250",
            "y": "145",
            "crop_size": "32",
            "camera_id": "net-camera-01",
            "mount_position": "NET_RIGHT",
            "zones": "FAR_RIGHT,NEAR_RIGHT",
            "receiving_side": "FAR",
            "depth_bu": "6",
        },
    )
    assert selected.status_code == 200, selected.text
    data = selected.json()
    assert data["configured"] is True
    assert data["next"] == "READY"
    assert registry.runtime is not None
    assert wizard_state.ball_template_path is not None
    assert wizard_state.ball_template_path.exists()


def test_reselect_ball_rebuilds_runtime_without_recalibrating_court():
    _reset()
    client = TestClient(app)
    bg = _court_rgb()
    client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png_bytes(bg), "image/png")},
        data={
            "image_points": "[[60,210],[260,210],[220,50],[100,50]]",
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    calibration_path = wizard_state.calibration_path

    for center, color in [((250,145),(250,245,20)), ((245,150),(40,250,80))]:
        f = bg.copy()
        cv2.circle(f, center, 6, color, -1)
        r = client.post(
            "/api/wizard/select-ball",
            files={"frame": ("ball.png", _png_bytes(f), "image/png")},
            data={
                "x": str(center[0]),
                "y": str(center[1]),
                "crop_size": "32",
                "camera_id": "net-camera-01",
                "mount_position": "NET_RIGHT",
                "zones": "FAR_RIGHT,NEAR_RIGHT",
                "receiving_side": "FAR",
                "depth_bu": "6",
            },
        )
        assert r.status_code == 200, r.text

    assert wizard_state.calibration_path == calibration_path
    assert registry.runtime is not None


def test_invalid_crossed_corner_order_is_rejected():
    _reset()
    client = TestClient(app)
    bg = _court_rgb()
    r = client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png_bytes(bg), "image/png")},
        data={
            "image_points": "[[60,210],[220,50],[260,210],[100,50]]",
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert r.status_code == 400


def test_documented_direct_launcher_can_import_linecaller():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/run_live_api.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout