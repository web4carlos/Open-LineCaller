from __future__ import annotations

import json
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from linecaller.api.boundary_calibration import (
    infer_court_corners_from_boundary_lines,
    parse_boundary_lines_json,
)
from linecaller.api.live_app import app, registry
from linecaller.api.live_wizard import wizard_state
from linecaller.dcf.external_grid_frame_loop import ExternalGridCalibration


SIZE = (320, 240)


def _background() -> np.ndarray:
    w, h = SIZE
    return np.full((h, w, 3), 30, dtype=np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _line_payload() -> dict:
    return {
        "NEAR": [[50, 220], [300, 220]],
        "RIGHT": [[300, 140], [250, 73.3333333333]],
        "FAR": [[100, 60], [240, 60]],
        "LEFT": [[50, 220], [100, 60]],
    }


def _reset() -> None:
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


def test_boundary_lines_infer_offscreen_near_right_corner():
    lines = parse_boundary_lines_json(json.dumps(_line_payload()), SIZE)
    result = infer_court_corners_from_boundary_lines(lines, SIZE)

    nl, nr, fr, fl = result.image_points
    assert nl == pytest.approx((50.0, 220.0), abs=0.2)
    assert nr[0] > SIZE[0]
    assert nr == pytest.approx((360.0, 220.0), abs=0.4)
    assert fr == pytest.approx((240.0, 60.0), abs=0.4)
    assert fl == pytest.approx((100.0, 60.0), abs=0.2)
    assert result.offscreen_corners == 1


def test_all_boundary_evidence_clicks_must_be_visible():
    payload = _line_payload()
    payload["RIGHT"][0] = [400, 140]
    with pytest.raises(ValueError, match="visible"):
        parse_boundary_lines_json(json.dumps(payload), SIZE)


def test_wizard_calibrates_from_lines_with_offscreen_corner():
    _reset()
    client = TestClient(app)

    response = client.post(
        "/api/wizard/calibrate",
        files={
            "background": (
                "court.png",
                _png_bytes(_background()),
                "image/png",
            )
        },
        data={
            "boundary_lines": json.dumps(_line_payload()),
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["calibration_method"] == "BOUNDARY_LINES"
    assert data["offscreen_corners"] == 1
    assert data["external_cells"] > 0
    assert data["image_points"][1][0] > SIZE[0]

    calibration = ExternalGridCalibration.load(wizard_state.calibration_path)
    assert calibration.image_points[1][0] > SIZE[0]

    w, h = SIZE
    for cell in calibration.cells:
        x0, y0, x1, y1 = cell.bbox_image
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h

        rx0, ry0, rx1, ry1 = cell.top_view_rect_bu
        court_x = calibration.config.court_x_bu
        court_y = calibration.config.full_court_y_bu
        assert not (
            rx0 >= 0.0 and rx1 <= court_x
            and ry0 >= 0.0 and ry1 <= court_y
        )


def test_existing_visible_corner_calibration_remains_compatible():
    _reset()
    client = TestClient(app)
    response = client.post(
        "/api/wizard/calibrate",
        files={
            "background": (
                "court.png",
                _png_bytes(_background()),
                "image/png",
            )
        },
        data={
            "image_points": "[[50,220],[300,220],[240,60],[100,60]]",
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["calibration_method"] == "VISIBLE_CORNERS"
    assert data["offscreen_corners"] == 0


def test_video_wizard_exposes_boundary_line_mode_and_ascii_step_labels():
    client = TestClient(app)
    text = client.get("/").text

    assert "Boundary Lines" in text
    assert "NEAR baseline" in text
    assert "RIGHT sideline" in text
    assert "FAR baseline" in text
    assert "LEFT sideline" in text
    assert "4 Visible Corners" in text
    assert "Step Frame Back" in text
    assert "Step Frame Forward" in text


def test_health_preserves_cp0036_2_and_reports_feature_version():
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["version"] == "CP-0036.2"
    assert health["feature_version"] == "CP-0036.2.1"