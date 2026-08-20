from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app

from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
)
from linecaller.dcf.projected_z0_identity import (
    ProjectedIdentityExternalGridFrameLoop,
    ProjectedZ0SignatureBank,
)


PTS = (
    (100.0, 500.0),
    (900.0, 500.0),
    (700.0, 100.0),
    (300.0, 100.0),
)
SIZE = (1000, 600)


def _ball_template() -> Image.Image:
    a = np.zeros((21, 21, 3), dtype=np.uint8)
    cv2.circle(a, (10, 10), 6, (245, 230, 40), -1, cv2.LINE_8)
    return Image.fromarray(a, "RGB")


def _background() -> Image.Image:
    return Image.new("RGB", SIZE, (20, 20, 20))


def _calibration():
    return ExternalGridCalibrator(
        PTS,
        SIZE,
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=6.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image="bg.png")


def _useful_signature(bank: ProjectedZ0SignatureBank):
    choices = []
    for cell_id in sorted(bank._by_cell):
        for signature in bank.signatures_for_cell(cell_id):
            if signature.anchor != "CENTER":
                continue
            x, y = signature.image_xy
            if (
                60 < x < SIZE[0] - 60
                and 80 < y < SIZE[1] - 60
                and signature.expected_diameter_px >= 4.0
            ):
                choices.append((y, signature))
    assert choices
    return max(choices, key=lambda item: item[0])[1]


def _paint(center, diameter) -> Image.Image:
    arr = np.asarray(_background(), dtype=np.uint8).copy()
    cx, cy = (int(round(center[0])), int(round(center[1])))
    r = max(2, int(round(float(diameter) / 2.0)))
    cv2.circle(arr, (cx, cy), r, (245, 230, 40), -1, cv2.LINE_8)
    return Image.fromarray(arr, "RGB")


def _strict_loop():
    cal = _calibration()
    profile = LockedBallColorProfile.from_template(_ball_template())
    loop = ProjectedIdentityExternalGridFrameLoop(
        cal,
        _background(),
        profile,
        difference_threshold=10,
        min_color_pixels=2,
        min_floor_scale_ratio=0.35,
        max_floor_scale_ratio=2.0,
        min_outside_clearance_bu=0.20,
        use_projected_signatures=True,
        require_approach_memory=True,
        approach_history_frames=3,
        approach_min_prior_observations=2,
        projection_anchor_distance_bu=0.95,
        approach_prediction_radius_bu=2.50,
        approach_min_total_motion_bu=0.65,
    )
    return cal, profile, loop


def test_every_external_cell_gets_five_projected_z0_ball_signatures():
    cal = _calibration()
    profile = LockedBallColorProfile.from_template(_ball_template())
    bank = ProjectedZ0SignatureBank(cal, profile)
    assert bank.signature_count == 5 * len(cal.cells)
    for cell in cal.cells[: min(25, len(cal.cells))]:
        anchors = {s.anchor for s in bank.signatures_for_cell(cell.cell_id)}
        assert anchors == {"CENTER", "NORTH", "SOUTH", "EAST", "WEST"}


def test_stationary_ball_colored_object_cannot_create_z0_with_approach_memory():
    cal, profile, loop = _strict_loop()
    bank = ProjectedZ0SignatureBank(cal, profile)
    signature = _useful_signature(bank)
    frame = _paint(signature.image_xy, signature.expected_diameter_px)

    r1 = loop.process_frame(1, frame)
    r2 = loop.process_frame(2, frame)
    r3 = loop.process_frame(3, frame)

    assert not r1.z0_candidates
    assert not r2.z0_candidates
    assert not r3.z0_candidates
    assert r3.projected_signature_matches >= 1
    assert r3.approach_rejections >= 1
    assert r3.predicted_z0_cell is None


def test_short_motion_chain_can_unlock_projected_z0_candidate():
    cal, profile, loop = _strict_loop()
    bank = ProjectedZ0SignatureBank(cal, profile)
    signature = _useful_signature(bank)

    cx, cy = signature.image_xy
    d = max(4.0, signature.expected_diameter_px)
    # Three sequential observations.  The first two are only image-space
    # history; the third lands on the projected exterior Z0 hypothesis.
    p1 = (cx - 2.4 * d, cy - 0.8 * d)
    p2 = (cx - 1.2 * d, cy - 0.4 * d)
    p3 = (cx, cy)

    loop.process_frame(10, _paint(p1, d))
    loop.process_frame(11, _paint(p2, d))
    result = loop.process_frame(12, _paint(p3, d))

    assert result.z0_candidates
    assert result.projected_signature_matches >= 1
    assert result.approach_rejections == 0
    assert result.motion_history_observations >= 2
    assert result.predicted_z0_cell == result.z0_candidates[0].cell_id
    assert "PROJECTED_Z0_SIGNATURE" in result.stage_trace
    assert "APPROACH_MEMORY" in result.stage_trace


def test_selected_ball_profile_persists_template_fill_identity():
    profile = LockedBallColorProfile.from_template(_ball_template())
    assert 0.30 <= profile.template_fill_ratio <= 1.0


def test_health_preserves_historical_runtime_key_and_adds_ball_identity_feature():
    data = TestClient(app).get("/health").json()
    assert data["runtime_feature_version"] == "CP-0036.2.3"
    assert data["ball_identity_feature_version"] == "CP-0036.2.4"
