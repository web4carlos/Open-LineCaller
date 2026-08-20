from __future__ import annotations

import json
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app, registry
from linecaller.api.live_wizard import wizard_state
from linecaller.dcf.external_grid_frame_loop import ExternalGridCalibration


SIZE = (320, 240)


def _png() -> bytes:
    image = np.full((SIZE[1], SIZE[0], 3), 35, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(image, "RGB").save(buf, format="PNG")
    return buf.getvalue()


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


def test_interactive_quad_accepts_reasonable_offscreen_corner():
    _reset()
    client = TestClient(app)
    quad = [[40, 220], [365, 220], [245, 62], [95, 62]]

    response = client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png(), "image/png")},
        data={
            "quad_points": json.dumps(quad),
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["calibration_method"] == "INTERACTIVE_QUAD"
    assert data["offscreen_corners"] == 1
    assert data["external_cells"] > 0

    calibration = ExternalGridCalibration.load(wizard_state.calibration_path)
    assert calibration.image_points[1][0] > SIZE[0]

    for cell in calibration.cells:
        x0, y0, x1, y1 = cell.top_view_rect_bu
        court_x = calibration.config.court_x_bu
        court_y = calibration.config.full_court_y_bu
        assert not (
            x0 >= 0.0 and x1 <= court_x
            and y0 >= 0.0 and y1 <= court_y
        )


def test_interactive_quad_rejects_implausibly_far_corner():
    _reset()
    client = TestClient(app)
    quad = [[40, 220], [5000, 220], [245, 62], [95, 62]]

    response = client.post(
        "/api/wizard/calibrate",
        files={"background": ("court.png", _png(), "image/png")},
        data={
            "quad_points": json.dumps(quad),
            "coverage": "FULL_COURT",
            "margin_bu": "6",
        },
    )
    assert response.status_code == 400
    assert "implausibly far outside" in response.json()["detail"]


def test_wizard_defaults_to_drag_handles_and_keeps_advanced_tools():
    text = TestClient(app).get("/").text
    assert "3. Adjust court" in text
    assert "Adjust 4 Corners" in text
    assert "Reset Quad" in text
    assert "Advanced calibration tools" in text
    assert "Boundary Lines" in text
    assert "4 Visible Corners" in text
    assert "Confirm Court & Build Outside Grid" in text


def test_wizard_has_pointer_drag_and_offscreen_handle_support():
    text = TestClient(app).get("/").text
    assert "pointerdown" in text
    assert "pointermove" in text
    assert "setPointerCapture" in text
    assert "handleDisplayPoint" in text
    assert " OFF" in text
    assert "quad_points" in text
    assert "INTERACTIVE_QUAD" in text


def test_historical_step_frame_wording_is_preserved():
    text = TestClient(app).get("/").text
    assert "Step Frame Back" in text
    assert "Step Frame Forward" in text
