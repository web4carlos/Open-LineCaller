from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.external_grid_pillow_watcher import (
    BallColorProfile,
    ExternalCell,
    ExternalGridCalibration,
    ExternalGridPillowWatcher,
)


def _profile_for_yellow():
    return BallColorProfile(
        hue=42.0,
        saturation=240.0,
        value=245.0,
        hue_tolerance=25.0,
        saturation_tolerance=100.0,
        value_tolerance=100.0,
    )


def _cal(tmp_path: Path):
    ref = np.zeros((80, 120, 3), dtype=np.uint8)
    ref[:] = (25, 25, 25)
    ref_path = tmp_path / "ref.png"
    Image.fromarray(ref, "RGB").save(ref_path)

    cells = [
        ExternalCell(
            cell_id=0,
            gx=-1,
            gy=0,
            polygon=((10, 20), (35, 20), (35, 55), (10, 55)),
            visible_area_px=26 * 36,
        ),
        ExternalCell(
            cell_id=1,
            gx=83,
            gy=0,
            polygon=((80, 20), (105, 20), (105, 55), (80, 55)),
            visible_area_px=26 * 36,
        ),
    ]
    return ExternalGridCalibration(
        image_size=(120, 80),
        mode="FULL_COURT",
        image_points=np.array([[20,70],[100,70],[75,20],[45,20]], dtype=np.float32),
        margin_bu=1,
        cells=cells,
        reference_image=str(ref_path),
    )


def _yellow_rgb():
    # PIL HSV hue around 42 for saturated yellow.
    return np.array([250, 245, 20], dtype=np.uint8)


def test_static_reference_has_no_candidate(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow())
    frame = np.asarray(Image.open(cal.reference_image).convert("RGB"))
    assert watcher.detect(frame) == []


def test_ball_like_change_inside_external_cell_raises_candidate(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow(), diff_threshold=10, min_ball_pixels=2)
    frame = np.asarray(Image.open(cal.reference_image).convert("RGB")).copy()
    frame[30:35, 20:25] = _yellow_rgb()
    candidates = watcher.detect(frame)
    assert candidates
    assert candidates[0].cell_id == 0


def test_same_stationary_ball_is_seen_again_because_reference_is_calibration_not_previous_frame(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow(), diff_threshold=10, min_ball_pixels=2)
    frame = np.asarray(Image.open(cal.reference_image).convert("RGB")).copy()
    frame[30:35, 20:25] = _yellow_rgb()
    first = watcher.detect(frame)
    second = watcher.detect(frame)
    assert first and second
    assert first[0].cell_id == second[0].cell_id == 0


def test_change_outside_external_cells_is_ignored(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow(), diff_threshold=10, min_ball_pixels=2)
    frame = np.asarray(Image.open(cal.reference_image).convert("RGB")).copy()
    frame[5:12, 55:62] = _yellow_rgb()
    assert watcher.detect(frame) == []


def test_non_ball_color_change_is_not_ball_candidate(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow(), diff_threshold=10, min_ball_pixels=2)
    frame = np.asarray(Image.open(cal.reference_image).convert("RGB")).copy()
    frame[30:40, 20:30] = np.array([30, 80, 250], dtype=np.uint8)
    assert watcher.detect(frame) == []


def test_calibration_build_creates_only_external_cells():
    points = [[0,70], [120,70], [85,15], [35,15]]
    cal = ExternalGridCalibration.build(
        image_size=(120,80),
        mode="FULL_COURT",
        image_points=points,
        margin_bu=2,
        reference_image="ref.png",
    )
    assert cal.cells
    assert all(not (0 <= c.gx < 83 and 0 <= c.gy < 183) for c in cal.cells)


def test_runtime_object_has_no_previous_frame_state(tmp_path):
    cal = _cal(tmp_path)
    watcher = ExternalGridPillowWatcher(cal, _profile_for_yellow())
    assert not hasattr(watcher, "previous_frame")
    assert not hasattr(watcher, "tracker")
    assert not hasattr(watcher, "trajectory")
