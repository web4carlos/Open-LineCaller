from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app
from linecaller.api.live_wizard import (
    _load_session_manifest,
    _restore_wizard_state_from_disk,
    _save_session_manifest,
    wizard_state,
)
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibrator,
    ExternalGridConfig,
)


def _persisted_artifacts(root):
    root.mkdir(parents=True, exist_ok=True)
    size = (320, 240)
    points = (
        (60.0, 210.0),
        (260.0, 210.0),
        (220.0, 50.0),
        (100.0, 50.0),
    )
    bg = Image.new("RGB", size, (25, 25, 25))
    bg_path = root / "EXTERNAL_GRID_BACKGROUND.png"
    bg.save(bg_path)

    cal = ExternalGridCalibrator(
        points,
        size,
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=6.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image=bg_path.name)
    cal.save(root / "EXTERNAL_GRID_CALIBRATION.json")

    ball = np.zeros((36, 36, 3), dtype=np.uint8)
    cv2.circle(ball, (18, 18), 7, (245, 230, 40), -1)
    Image.fromarray(ball, "RGB").save(root / "BALL_TEMPLATE.png")
    return cal


def test_legacy_three_artifact_session_can_restore_without_recalibration(tmp_path):
    wizard_dir = tmp_path / "wizard"
    cal = _persisted_artifacts(wizard_dir)
    wizard_state.reset()

    assert _restore_wizard_state_from_disk(wizard_dir) is True
    assert wizard_state.calibration_path is not None
    assert wizard_state.background_path is not None
    assert wizard_state.ball_template_path is not None
    assert wizard_state.external_cells == len(cal.cells)
    assert wizard_state.restore_available is True
    assert wizard_state.persisted_session is False


def test_manifest_persists_ownership_for_future_automatic_boots(tmp_path):
    wizard_dir = tmp_path / "wizard"
    _persisted_artifacts(wizard_dir)
    path = _save_session_manifest(
        wizard_dir,
        camera_id="video-camera-01",
        mount_position="NET_CENTER",
        zones=["FAR_LEFT", "FAR_BASELINE"],
        receiving_side="FAR",
        depth_bu=6.0,
    )
    assert path.exists()
    manifest = _load_session_manifest(wizard_dir)
    assert manifest is not None
    assert manifest["camera_id"] == "video-camera-01"
    assert manifest["mount_position"] == "NET_CENTER"
    assert manifest["zones"] == ["FAR_LEFT", "FAR_BASELINE"]
    assert manifest["receiving_side"] == "FAR"
    assert manifest["depth_bu"] == 6.0

    wizard_state.reset()
    assert _restore_wizard_state_from_disk(wizard_dir) is True
    assert wizard_state.persisted_session is True


def test_wizard_exposes_resume_and_preserves_runtime_when_video_is_loaded():
    html = TestClient(app).get("/").text
    assert 'id="resumeSession"' in html
    assert "Resume Previous Calibration" in html
    assert "fetch('/api/wizard/restore'" in html
    assert "const hadConfiguredRuntime = configured;" in html
    assert "if (hadConfiguredRuntime)" in html
    assert "checkPersistentSession();" in html
    assert "you do NOT need to recalibrate" in html


def test_wizard_status_exposes_persistence_state():
    data = TestClient(app).get("/api/wizard/status").json()
    wizard = data["wizard"]
    assert "restore_available" in wizard
    assert "persisted_session" in wizard
    assert "restore_error" in wizard


def test_health_advertises_persistent_wizard_session_restore():
    data = TestClient(app).get("/health").json()
    assert data["wizard_session_feature_version"] == "CP-0036.2.4.3"
