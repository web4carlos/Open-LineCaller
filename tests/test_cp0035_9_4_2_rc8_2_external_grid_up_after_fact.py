import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibrator,
    ExternalGridConfig,
    ExternalGridFrameLoop,
    LockedBallColorProfile,
    ExternalGridCalibration,
)

PTS = ((100.0, 500.0), (900.0, 500.0), (700.0, 100.0), (300.0, 100.0))
SIZE = (1000, 600)


def make_ball_template():
    a = np.zeros((15, 15, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:15, :15]
    mask = (xx - 7) ** 2 + (yy - 7) ** 2 <= 5 ** 2
    a[mask] = (245, 230, 40)
    return Image.fromarray(a, "RGB")


def build(coverage="FULL_COURT"):
    cfg = ExternalGridConfig(margin_bu=4.0)
    return ExternalGridCalibrator(
        PTS, SIZE, coverage=coverage, config=cfg, image_up_unit=(0.0, -1.0)
    ).build(background_image="bg.png")


def paint_ball(base, center, diameter, color=(245, 230, 40)):
    arr = np.asarray(base).copy()
    cx, cy = center
    r = max(2, int(round(diameter / 2)))
    yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    arr[mask] = color
    return Image.fromarray(arr, "RGB")


def useful_cell(cal):
    # Prefer a near external cell with room above it in the image.
    candidates = []
    for c in cal.cells:
        poly = np.asarray(c.polygon_image, dtype=float)
        cx, cy = poly.mean(axis=0)
        d = c.expected_floor_ball_diameter_px
        if 40 < cx < SIZE[0] - 40 and 80 < cy < SIZE[1] - 40 and d >= 4:
            candidates.append((cy, c))
    assert candidates
    return max(candidates, key=lambda item: item[0])[1]


def center_of(cell):
    poly = np.asarray(cell.polygon_image, dtype=float)
    return tuple(poly.mean(axis=0))


def test_calibration_builds_external_grid_only():
    cal = build()
    assert cal.cells
    assert all(c.region.startswith("OUT_") for c in cal.cells)


def test_top_view_grid_is_projected_into_perspective():
    cal = build()
    # Court coordinates: FAR is low Y, NEAR is high Y.
    far = [c for c in cal.cells if c.region == "OUT_LEFT" and c.iy == 5]
    near = [c for c in cal.cells if c.region == "OUT_LEFT" and c.iy == 175]
    assert near and far
    assert near[0].expected_floor_ball_diameter_px > far[0].expected_floor_ball_diameter_px


def test_full_and_half_court_supported_by_calibration():
    full = build("FULL_COURT")
    half = build("HALF_COURT")
    assert full.coverage == CalibrationCoverage.FULL_COURT.value
    assert half.coverage == CalibrationCoverage.HALF_COURT.value


def test_calibration_persists_image_up_direction():
    cal = build()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "grid.json"
        cal.save(p)
        loaded = ExternalGridCalibration.load(p)
    assert loaded.image_up_unit == (0.0, -1.0)


def test_runtime_order_starts_frame_then_pillow_then_external_loop():
    cal = build()
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()))
    result = loop.process_frame(1, bg)
    assert result.stage_trace[:3] == ("FRAME", "PILLOW", "EXTERNAL_CELL_LOOP")


def test_z0_candidate_alone_is_not_bingo():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    contact = paint_ball(bg, center_of(cell), cell.expected_floor_ball_diameter_px)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(9, bg)
    result = loop.process_frame(10, contact)
    assert result.z0_candidates
    assert not result.bingo_cells


def test_same_ball_one_ball_up_after_fact_is_bingo():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    contact = paint_ball(bg, (cx, cy), d)
    above = paint_ball(bg, (cx, cy - 1.05 * d), d)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(19, bg)
    r0 = loop.process_frame(20, contact)
    r1 = loop.process_frame(21, above)
    assert r0.z0_candidates and not r0.bingo_cells
    assert r1.bingo_cells
    assert r1.bingo_cells[0].up_bu > 0


def test_no_down_prerequisite_exists():
    cal = build()
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()))
    assert not hasattr(loop, "down")
    assert not hasattr(loop, "descending")
    assert not hasattr(loop, "velocity")
    assert not hasattr(loop, "trajectory")
    assert not hasattr(loop, "ball_position")


def test_movement_below_contact_does_not_confirm_up():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(29, bg)
    loop.process_frame(30, paint_ball(bg, (cx, cy), d))
    r = loop.process_frame(31, paint_ball(bg, (cx, cy + 1.1 * d), d))
    assert not r.bingo_cells


def test_small_up_noise_is_not_enough():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(39, bg)
    loop.process_frame(40, paint_ball(bg, (cx, cy), d))
    r = loop.process_frame(41, paint_ball(bg, (cx, cy - 0.25 * d), d))
    assert not r.bingo_cells


def test_wrong_color_above_is_not_same_ball():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(49, bg)
    loop.process_frame(50, paint_ball(bg, (cx, cy), d))
    r = loop.process_frame(51, paint_ball(bg, (cx, cy - 1.1 * d), d, color=(40, 80, 245)))
    assert not r.bingo_cells


def test_up_confirmation_can_arrive_a_few_frames_later():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2, max_after_frames=3)
    loop.process_frame(59, bg)
    loop.process_frame(60, paint_ball(bg, (cx, cy), d))
    assert not loop.process_frame(61, bg).bingo_cells
    r = loop.process_frame(62, paint_ball(bg, (cx, cy - 1.2 * d), d))
    assert r.bingo_cells
    assert r.bingo_cells[0].confirm_frame == 62


def test_pending_candidate_expires():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    cx, cy = center_of(cell)
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2, max_after_frames=2)
    loop.process_frame(69, bg)
    loop.process_frame(70, paint_ball(bg, (cx, cy), d))
    loop.process_frame(71, bg)
    loop.process_frame(72, bg)
    r = loop.process_frame(73, paint_ball(bg, (cx, cy - 1.1 * d), d))
    assert not r.bingo_cells


def test_single_frame_can_never_be_bingo():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2)
    loop.process_frame(79, bg)
    r = loop.process_frame(80, paint_ball(bg, center_of(cell), d))
    assert not r.bingo_cells


def test_pillow_uses_calibration_reference_not_previous_frame():
    cal = build()
    cell = useful_cell(cal)
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    d = cell.expected_floor_ball_diameter_px
    contact = paint_ball(bg, center_of(cell), d)
    loop = ExternalGridFrameLoop(
        cal, bg, LockedBallColorProfile.from_template(make_ball_template()), min_color_pixels=2
    )
    first = loop.process_frame(100, contact)
    second = loop.process_frame(101, contact)
    assert first.z0_candidates
    # If runtime compared against previous frame this would disappear here.
    assert second.z0_candidates


def test_runtime_has_no_previous_frame_reference_state():
    cal = build()
    bg = Image.new("RGB", SIZE, (20, 20, 20))
    loop = ExternalGridFrameLoop(cal, bg, LockedBallColorProfile.from_template(make_ball_template()))
    assert not hasattr(loop, "_previous_frame")


def test_calibrated_court_corners_follow_linecaller_y_convention():
    builder = ExternalGridCalibrator(PTS, SIZE, coverage="FULL_COURT", config=ExternalGridConfig(margin_bu=4.0))
    y = builder.config.full_court_y_bu
    checks = [
        (builder.project_top_view(0.0, y), PTS[0]),
        (builder.project_top_view(builder.config.court_x_bu, y), PTS[1]),
        (builder.project_top_view(builder.config.court_x_bu, 0.0), PTS[2]),
        (builder.project_top_view(0.0, 0.0), PTS[3]),
    ]
    for got, expected in checks:
        assert np.allclose(got, expected, atol=1e-3)


def test_projected_cell_is_quadrilateral_not_axis_aligned_image_box():
    cal = build()
    candidates = [c for c in cal.cells if c.region == "OUT_LEFT" and 20 < c.iy < 160]
    assert candidates
    c = candidates[len(candidates)//2]
    p = np.asarray(c.polygon_image)
    # Opposing floor-grid rows are generally not horizontal after perspective.
    # Most importantly the stored watch geometry is four projected vertices,
    # not bbox_image.
    assert p.shape == (4, 2)
    assert tuple(map(tuple, p)) != ((c.bbox_image[0], c.bbox_image[1]), (c.bbox_image[2], c.bbox_image[1]), (c.bbox_image[2], c.bbox_image[3]), (c.bbox_image[0], c.bbox_image[3]))
