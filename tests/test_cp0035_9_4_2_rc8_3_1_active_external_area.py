from __future__ import annotations

import numpy as np

from linecaller.dcf.active_external_area import (
    calibration_for_active_area,
    select_active_external_area,
)
from linecaller.dcf.external_grid_pillow_watcher import (
    ExternalCell,
    ExternalGridCalibration,
)


def _cell(cid, gx, gy):
    return ExternalCell(
        cell_id=cid,
        gx=gx,
        gy=gy,
        polygon=((0,0),(1,0),(1,1),(0,1)),
        visible_area_px=4,
    )


def _full_cal(cells):
    return ExternalGridCalibration(
        image_size=(640,360),
        mode="FULL_COURT",
        image_points=np.array([[0,345],[666,345],[390,170],[257,170]], dtype=np.float32),
        margin_bu=12,
        cells=cells,
        reference_image="reference.png",
    )


def test_far_selects_far_baseline_band():
    cal = _full_cal([_cell(1, 40, -1), _cell(2, 40, 183)])
    area = select_active_external_area(cal, "FAR")
    assert [c.cell_id for c in area.cells] == [1]


def test_near_selects_near_baseline_band():
    cal = _full_cal([_cell(1, 40, -1), _cell(2, 40, 183)])
    area = select_active_external_area(cal, "NEAR")
    assert [c.cell_id for c in area.cells] == [2]


def test_far_selects_only_sideline_adjacent_to_far_half():
    cal = _full_cal([
        _cell(1, -1, 20),
        _cell(2, -1, 150),
        _cell(3, 83, 30),
        _cell(4, 83, 140),
    ])
    area = select_active_external_area(cal, "FAR")
    assert {c.cell_id for c in area.cells} == {1, 3}


def test_near_selects_only_sideline_adjacent_to_near_half():
    cal = _full_cal([
        _cell(1, -1, 20),
        _cell(2, -1, 150),
        _cell(3, 83, 30),
        _cell(4, 83, 140),
    ])
    area = select_active_external_area(cal, "NEAR")
    assert {c.cell_id for c in area.cells} == {2, 4}


def test_area_reduces_full_external_loop():
    cal = _full_cal([
        _cell(1, 40, -1),
        _cell(2, 40, 183),
        _cell(3, -1, 20),
        _cell(4, -1, 150),
        _cell(5, 83, 30),
        _cell(6, 83, 140),
    ])
    area = select_active_external_area(cal, "FAR")
    assert area.active_cell_count < area.full_external_cell_count
    assert area.reduction_ratio > 0.0


def test_runtime_view_keeps_same_geometry_and_reference():
    cal = _full_cal([_cell(1, 40, -1), _cell(2, 40, 183)])
    area = select_active_external_area(cal, "FAR")
    view = calibration_for_active_area(cal, area)
    assert view.reference_image == cal.reference_image
    assert np.array_equal(view.image_points, cal.image_points)
    assert len(view.cells) == 1
    assert view.cells[0].cell_id == 1


def test_invalid_side_rejected():
    cal = _full_cal([_cell(1, 40, -1)])
    try:
        select_active_external_area(cal, "LEFT")  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
