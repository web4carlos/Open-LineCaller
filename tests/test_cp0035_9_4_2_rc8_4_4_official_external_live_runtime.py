from __future__ import annotations

import cv2
import numpy as np

from linecaller.dcf.camera_external_ownership import CameraExternalOwnership
from linecaller.dcf.camera_pose_guard import CameraPoseGuardConfig, PoseGuardState
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridConfig,
    LockedBallColorProfile,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
    OfficialExternalLiveRuntime,
)


W, H = 320, 240
COURT = (
    (60.0, 210.0),
    (260.0, 210.0),
    (220.0, 50.0),
    (100.0, 50.0),
)


def background():
    im = np.full((H, W, 3), 28, dtype=np.uint8)
    p = np.rint(np.asarray(COURT)).astype(np.int32)
    cv2.polylines(im, [p.reshape(-1, 1, 2)], True, (225,225,225), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate([
        (60,210),(80,210),(110,210),(145,210),(180,210),(215,210),(245,210),(260,210),
        (100,50),(125,50),(155,50),(185,50),(220,50),
        (67,185),(74,160),(82,135),(89,110),(95,80),
        (253,185),(246,160),(239,135),(232,110),(225,80),
    ]):
        r = 3 + (i % 2)
        cv2.rectangle(im, (x-r,y-r), (x+r,y+r), (245,245,245), -1)
        cv2.line(im, (x-r-2,y), (x+r+2,y), (70,70,70), 1)
        cv2.line(im, (x,y-r-2), (x,y+r+2), (70,70,70), 1)
    return im


def cell(cid, ix, iy, region, rect, poly, expected=10.0):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return ExternalGridCell(
        cell_id=cid,
        ix=ix,
        iy=iy,
        region=region,
        top_view_rect_bu=rect,
        polygon_image=tuple(poly),
        bbox_image=(int(min(xs))-1,int(min(ys))-1,int(max(xs))+2,int(max(ys))+2),
        expected_floor_ball_diameter_px=float(expected),
    )


FAR_RIGHT = cell(1,83,30,"OUT_RIGHT",(83.0,30.0,84.0,31.0),((244.0,140.0),(258.0,140.0),(258.0,154.0),(244.0,154.0)),10.0)
FAR_LEFT = cell(2,-1,30,"OUT_LEFT",(-1.0,30.0,0.0,31.0),((62.0,140.0),(76.0,140.0),(76.0,154.0),(62.0,154.0)),10.0)
NEAR_RIGHT = cell(3,83,150,"OUT_RIGHT",(83.0,150.0,84.0,151.0),((258.0,180.0),(272.0,180.0),(272.0,194.0),(258.0,194.0)),10.0)


def calibration():
    return ExternalGridCalibration(
        version=2,
        coverage="FULL_COURT",
        image_size=(W,H),
        image_points=COURT,
        config=ExternalGridConfig(court_x_bu=83.0, full_court_y_bu=182.0, half_court_y_bu=91.0, cell_bu=1.0, margin_bu=10.0),
        background_image="reference.png",
        image_up_unit=(0.0,-1.0),
        cells=(FAR_RIGHT,FAR_LEFT,NEAR_RIGHT),
    )


def ball_profile():
    return LockedBallColorProfile(hue_center=42.0, hue_tolerance=22.0, saturation_min=120, value_min=120)


def pose_config():
    return CameraPoseGuardConfig(
        anchor_band_px=30,
        max_features=160,
        min_tracks=10,
        safe_max_corner_shift_px=2.5,
        safe_rmse_px=1.6,
        micro_adjust_max_corner_shift_px=12.0,
        micro_adjust_max_rmse_px=3.5,
    )


def live_config():
    return OfficialExternalLiveConfig(
        glow_hold_s=2.0,
        difference_threshold=12,
        min_color_pixels=4,
        min_floor_scale_ratio=0.40,
        max_floor_scale_ratio=1.60,
        min_up_bu=0.55,
        max_up_bu=2.20,
        max_lateral_bu=1.10,
        min_after_scale_ratio=0.40,
        max_after_scale_ratio=1.80,
        max_after_frames=2,
    )


def ownership():
    return CameraExternalOwnership(
        camera_id="net-right-01",
        mount_position="NET_RIGHT",
        zones=("FAR_RIGHT","NEAR_RIGHT"),
        depth_bu=4,
    )


def runtime():
    bg = background()
    return OfficialExternalLiveRuntime(
        calibration(), bg, ball_profile(), ownership(),
        receiving_side="FAR",
        config=live_config(),
        pose_config=pose_config(),
    ), bg


def yellow():
    return np.array([250,245,20], dtype=np.uint8)


def draw_ball(frame, cx, cy, r=4):
    out = frame.copy()
    cv2.circle(out, (int(cx),int(cy)), int(r), tuple(int(v) for v in yellow()), -1, cv2.LINE_AA)
    return out


def warp_translation(image, dx, dy):
    M = np.array([[1.0,0.0,float(dx)],[0.0,1.0,float(dy)],[0.0,0.0,1.0]], dtype=np.float64)
    return cv2.warpPerspective(image, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_runtime_scans_only_camera_owned_receiving_side_cells():
    rt, _ = runtime()
    assert rt.active_external_cell_count == 1


def test_z0_candidate_alone_never_produces_out():
    rt, bg = runtime()
    candidate = draw_ball(bg, 251, 147)
    result = rt.process_frame(10, candidate, now_s=1.0)
    assert result.pose.state is PoseGuardState.SAFE
    assert not result.scan_suppressed
    assert result.frame_loop_result is not None
    assert result.frame_loop_result.z0_candidates
    assert result.out_calls == ()


def test_candidate_then_up_after_fact_produces_one_official_out_and_glow():
    rt, bg = runtime()
    first = rt.process_frame(10, draw_ball(bg,251,147), now_s=1.0)
    assert first.out_calls == ()
    above = draw_ball(bg,251,137)
    second = rt.process_frame(11, above, now_s=1.05)
    assert second.pose.state is PoseGuardState.SAFE
    assert second.has_out
    assert len(second.out_calls) == 1
    call = second.out_calls[0]
    assert call.call == "OUT"
    assert call.cell_id == FAR_RIGHT.cell_id
    assert call.region == "OUT_RIGHT"
    assert call.camera_id == "net-right-01"
    assert call.receiving_side == "FAR"
    assert rt.active_glows
    assert int(second.rendered_bgr.sum()) > int(above[:,:,::-1].sum())


def test_unowned_external_cell_cannot_create_candidate_or_out():
    rt, bg = runtime()
    result = rt.process_frame(10, draw_ball(bg,69,147), now_s=1.0)
    assert result.pose.state is PoseGuardState.SAFE
    assert result.frame_loop_result is not None
    assert result.frame_loop_result.z0_candidates == ()
    assert result.out_calls == ()


def test_receiving_side_switch_rebuilds_owned_external_area_and_clears_event_state():
    rt, bg = runtime()
    rt.process_frame(10, draw_ball(bg,251,147), now_s=1.0)
    rt.set_receiving_side("NEAR")
    assert rt.receiving_side == "NEAR"
    assert rt.active_external_cell_count == 1
    assert rt.active_glows == ()
    result = rt.process_frame(11, draw_ball(bg,251,137), now_s=1.05)
    assert result.out_calls == ()


def test_large_camera_shift_blocks_before_external_grid_scan():
    rt, bg = runtime()
    result = rt.process_frame(10, warp_translation(bg,24,0), now_s=1.0)
    assert result.pose.state is PoseGuardState.BLOCKED
    assert result.scan_suppressed
    assert result.frame_loop_result is None
    assert result.out_calls == ()
    assert result.reason == "POSE_BLOCKED_NO_CALL"


def test_micro_adjust_repairs_pose_but_suppresses_transition_frame():
    rt, bg = runtime()
    shifted = warp_translation(bg,6,0)
    transition = rt.process_frame(10, shifted, now_s=1.0)
    assert transition.pose.state is PoseGuardState.MICRO_ADJUST
    assert transition.scan_suppressed
    assert transition.frame_loop_result is None
    assert transition.out_calls == ()
    assert transition.reason == "POSE_REPAIRED_RECHECK_REQUIRED"
    assert rt.pose_epoch == 1
    stable = rt.process_frame(11, shifted, now_s=1.05)
    assert stable.pose.state is PoseGuardState.SAFE
    assert not stable.scan_suppressed
    assert stable.out_calls == ()


def test_no_current_side_ownership_fails_closed_with_zero_scan():
    bg = background()
    far_only = CameraExternalOwnership(
        camera_id="far-only",
        mount_position="NET_RIGHT",
        zones=("FAR_RIGHT",),
        depth_bu=4,
    )
    rt = OfficialExternalLiveRuntime(
        calibration(), bg, ball_profile(), far_only,
        receiving_side="NEAR",
        config=live_config(),
        pose_config=pose_config(),
    )
    assert rt.active_external_cell_count == 0
    result = rt.process_frame(1, bg, now_s=0.0)
    assert result.pose.state is PoseGuardState.SAFE
    assert result.scan_suppressed
    assert result.frame_loop_result is None
    assert result.out_calls == ()
    assert result.reason == "NO_CAMERA_OWNERSHIP_FOR_RECEIVING_SIDE"