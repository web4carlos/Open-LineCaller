from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import pytest

from linecaller.dcf.camera_pose_guard import (
    CameraPoseGuard,
    CameraPoseGuardConfig,
    PoseGuardState,
    apply_micro_adjustment,
)
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridConfig,
)


W, H = 320, 240
COURT = (
    (60.0, 210.0),
    (260.0, 210.0),
    (220.0, 50.0),
    (100.0, 50.0),
)


def _background():
    im = np.full((H, W, 3), 28, dtype=np.uint8)
    p = np.rint(np.asarray(COURT)).astype(np.int32)
    cv2.polylines(im, [p.reshape(-1, 1, 2)], True, (225, 225, 225), 3, cv2.LINE_AA)

    # Static high-contrast texture near calibrated boundary anchors.
    for i, (x, y) in enumerate(
        [
            (60, 210), (80, 210), (110, 210), (145, 210), (180, 210),
            (215, 210), (245, 210), (260, 210),
            (100, 50), (125, 50), (155, 50), (185, 50), (220, 50),
            (67, 185), (74, 160), (82, 135), (89, 110), (95, 80),
            (253, 185), (246, 160), (239, 135), (232, 110), (225, 80),
        ]
    ):
        r = 3 + (i % 2)
        cv2.rectangle(im, (x-r, y-r), (x+r, y+r), (245, 245, 245), -1)
        cv2.line(im, (x-r-2, y), (x+r+2, y), (70, 70, 70), 1)
        cv2.line(im, (x, y-r-2), (x, y+r+2), (70, 70, 70), 1)
    return im


def _cell():
    return ExternalGridCell(
        cell_id=7,
        ix=-1,
        iy=40,
        region="OUT_LEFT",
        top_view_rect_bu=(-1.0, 40.0, 0.0, 41.0),
        polygon_image=((53.0, 176.0), (60.0, 176.0), (61.0, 171.0), (54.0, 171.0)),
        bbox_image=(52, 170, 63, 178),
        expected_floor_ball_diameter_px=7.0,
    )


def _cal():
    return ExternalGridCalibration(
        version=2,
        coverage="FULL_COURT",
        image_size=(W, H),
        image_points=COURT,
        config=ExternalGridConfig(
            court_x_bu=83.0,
            full_court_y_bu=182.0,
            half_court_y_bu=91.0,
            cell_bu=1.0,
            margin_bu=10.0,
        ),
        background_image="reference.png",
        image_up_unit=(0.0, -1.0),
        cells=(_cell(),),
    )


def _cfg():
    return CameraPoseGuardConfig(
        anchor_band_px=30,
        max_features=160,
        min_tracks=10,
        safe_max_corner_shift_px=2.5,
        safe_rmse_px=1.6,
        micro_adjust_max_corner_shift_px=12.0,
        micro_adjust_max_rmse_px=3.5,
    )


def _warp_translation(image, dx, dy):
    M = np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    out = cv2.warpPerspective(
        image,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out


def test_stable_reference_allows_out_call():
    bg = _background()
    guard = CameraPoseGuard(_cal(), bg, config=_cfg())

    assert guard.reference_feature_count >= 10
    d = guard.evaluate(bg)

    assert d.state is PoseGuardState.SAFE
    assert d.allows_out_call
    assert not d.requires_micro_adjustment
    assert d.max_corner_shift_px is not None
    assert d.max_corner_shift_px <= 1.0


def test_small_measurable_shift_requires_micro_adjustment_not_out_call():
    bg = _background()
    guard = CameraPoseGuard(_cal(), bg, config=_cfg())
    current = _warp_translation(bg, 6, 0)

    d = guard.evaluate(current)

    assert d.state is PoseGuardState.MICRO_ADJUST
    assert not d.allows_out_call
    assert d.requires_micro_adjustment
    assert d.homography_ref_to_current is not None


def test_large_camera_shift_blocks_officiating():
    bg = _background()
    guard = CameraPoseGuard(_cal(), bg, config=_cfg())
    current = _warp_translation(bg, 24, 0)

    d = guard.evaluate(current)

    assert d.state is PoseGuardState.BLOCKED
    assert d.blocked
    assert not d.allows_out_call


def test_missing_anchor_evidence_fails_closed():
    bg = _background()
    guard = CameraPoseGuard(_cal(), bg, config=_cfg())
    blank = np.zeros_like(bg)

    d = guard.evaluate(blank)

    assert d.state is PoseGuardState.BLOCKED
    assert not d.allows_out_call


def test_micro_adjustment_moves_grid_and_background_together():
    bg = _background()
    cal = _cal()
    guard = CameraPoseGuard(cal, bg, config=_cfg())
    current = _warp_translation(bg, 6, 0)
    d = guard.evaluate(current)

    assert d.state is PoseGuardState.MICRO_ADJUST
    adjusted = apply_micro_adjustment(cal, Image.fromarray(bg, "RGB"), d)

    # World/exterior identity is immutable.
    assert len(adjusted.calibration.cells) == len(cal.cells)
    assert adjusted.calibration.cells[0].cell_id == cal.cells[0].cell_id
    assert adjusted.calibration.cells[0].region == "OUT_LEFT"
    assert adjusted.calibration.cells[0].top_view_rect_bu == cal.cells[0].top_view_rect_bu

    # Image geometry follows the moved camera.
    old_x = np.mean([p[0] for p in cal.cells[0].polygon_image])
    new_x = np.mean([p[0] for p in adjusted.calibration.cells[0].polygon_image])
    assert new_x - old_x == pytest.approx(6.0, abs=1.5)

    # The calibration-owned reference is transformed by the same pose repair.
    warped = np.asarray(adjusted.background_rgb, dtype=np.uint8)
    assert warped.shape == bg.shape


def test_after_micro_adjustment_new_guard_sees_pose_as_stable():
    bg = _background()
    cal = _cal()
    guard = CameraPoseGuard(cal, bg, config=_cfg())
    current = _warp_translation(bg, 6, 0)
    d = guard.evaluate(current)
    adjusted = apply_micro_adjustment(cal, bg, d)

    # A new guard is created on the repaired reference/calibration.
    repaired_guard = CameraPoseGuard(
        adjusted.calibration,
        adjusted.background_rgb,
        config=_cfg(),
    )
    repaired = repaired_guard.evaluate(current)

    assert repaired.state is PoseGuardState.SAFE
    assert repaired.allows_out_call