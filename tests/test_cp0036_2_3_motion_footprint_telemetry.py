from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from linecaller.api.live_app import app
from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridConfig,
    ExternalGridFrameLoop,
    LockedBallColorProfile,
)


def _profile() -> LockedBallColorProfile:
    return LockedBallColorProfile(
        hue_center=42.0,
        hue_tolerance=24.0,
        saturation_min=100,
        value_min=100,
    )


def _simple_calibration(
    *,
    coverage: str = "FULL_COURT",
    expected_px: float = 8.0,
) -> ExternalGridCalibration:
    w, h = 320, 240
    cfg = ExternalGridConfig(
        court_x_bu=83.0,
        full_court_y_bu=182.0,
        half_court_y_bu=91.0,
        cell_bu=1.0,
        margin_bu=10.0,
    )
    cell = ExternalGridCell(
        cell_id=7,
        ix=83,
        iy=30,
        region="OUT_RIGHT",
        top_view_rect_bu=(83.0, 30.0, 84.0, 31.0),
        polygon_image=((250.0, 120.0), (290.0, 120.0), (290.0, 170.0), (250.0, 170.0)),
        bbox_image=(249, 119, 291, 171),
        expected_floor_ball_diameter_px=expected_px,
    )
    return ExternalGridCalibration(
        version=2,
        coverage=coverage,
        image_size=(w, h),
        image_points=((50.0, 200.0), (250.0, 200.0), (250.0, 40.0), (50.0, 40.0)),
        config=cfg,
        background_image="reference.png",
        image_up_unit=(0.0, -1.0),
        cells=(cell,),
    )


def _background() -> Image.Image:
    return Image.fromarray(np.full((240, 320, 3), 28, dtype=np.uint8), "RGB")


def _yellow_rgb() -> tuple[int, int, int]:
    return (250, 245, 20)


def test_core_plus_halo_are_merged_into_one_ball_motion_footprint():
    bg = np.asarray(_background(), dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (268, 145), 3, _yellow_rgb(), -1, cv2.LINE_8)
    cv2.circle(frame, (278, 145), 3, _yellow_rgb(), -1, cv2.LINE_8)

    loop = ExternalGridFrameLoop(
        _simple_calibration(expected_px=7.0),
        _background(),
        _profile(),
        difference_threshold=10,
        min_color_pixels=3,
        min_floor_scale_ratio=0.25,
        max_floor_scale_ratio=2.0,
        motion_merge_gap_px=8.0,
        min_outside_clearance_bu=0.0,
    )
    result = loop.process_frame(10, frame)

    assert result.raw_ball_components == 2
    assert result.ball_footprints == 1
    assert result.merged_motion_footprints == 1


def test_line_touch_guard_rejects_boundary_contact_before_z0():
    bg = np.asarray(_background(), dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (250, 145), 3, _yellow_rgb(), -1, cv2.LINE_8)

    loop = ExternalGridFrameLoop(
        _simple_calibration(expected_px=7.0),
        _background(),
        _profile(),
        difference_threshold=10,
        min_color_pixels=3,
        min_floor_scale_ratio=0.25,
        max_floor_scale_ratio=2.0,
        min_outside_clearance_bu=0.20,
    )
    result = loop.process_frame(20, frame)

    assert result.boundary_guard_rejections >= 1
    assert result.z0_candidates == ()
    assert result.up_confirmations == ()


def test_clear_external_contact_can_still_become_z0():
    bg = np.asarray(_background(), dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (270, 145), 3, _yellow_rgb(), -1, cv2.LINE_8)

    loop = ExternalGridFrameLoop(
        _simple_calibration(expected_px=7.0),
        _background(),
        _profile(),
        difference_threshold=10,
        min_color_pixels=3,
        min_floor_scale_ratio=0.25,
        max_floor_scale_ratio=2.0,
        min_outside_clearance_bu=0.20,
    )
    result = loop.process_frame(21, frame)

    assert result.boundary_guard_rejections == 0
    assert result.z0_candidates
    assert result.z0_candidates[0].boundary_clearance_bu > 0.20


def test_half_court_uses_same_boundary_guard_geometry():
    cal = _simple_calibration(coverage="HALF_COURT", expected_px=7.0)
    assert CalibrationCoverage.parse(cal.coverage) is CalibrationCoverage.HALF_COURT

    bg = np.asarray(_background(), dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (250, 145), 3, _yellow_rgb(), -1, cv2.LINE_8)

    loop = ExternalGridFrameLoop(
        cal,
        _background(),
        _profile(),
        difference_threshold=10,
        min_color_pixels=3,
        min_floor_scale_ratio=0.25,
        max_floor_scale_ratio=2.0,
        min_outside_clearance_bu=0.20,
    )
    result = loop.process_frame(22, frame)
    assert result.boundary_guard_rejections >= 1
    assert result.z0_candidates == ()


def test_live_api_exposes_runtime_feature_and_telemetry_ui():
    client = TestClient(app)
    health = client.get("/health").json()

    assert health["version"] == "CP-0036.2"
    assert health["feature_version"] == "CP-0036.2.1"
    assert health["runtime_feature_version"] == "CP-0036.2.3"

    text = client.get("/").text
    assert "Live Event Telemetry" in text
    assert "Ball pixels" in text
    assert "Raw blobs" in text
    assert "Footprints" in text
    assert "Blur merges" in text
    assert "Z0 total" in text
    assert "UP total" in text
    assert "OUT total" in text
    assert "Line guards" in text
    assert "/api/live/reset-events" in text
