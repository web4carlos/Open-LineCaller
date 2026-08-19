from __future__ import annotations

import numpy as np
import pytest

from linecaller.dcf.external_grid_pillow_watcher import (
    ExternalCell,
    ExternalGridCalibration,
)
from linecaller.dcf.specific_external_roi import (
    calibration_for_specific_roi,
    select_specific_external_roi,
)


def _cell(cid, gx, gy):
    return ExternalCell(
        cell_id=cid,
        gx=gx,
        gy=gy,
        polygon=((0,0),(1,0),(1,1),(0,1)),
        visible_area_px=4,
    )


def _cal(cells):
    return ExternalGridCalibration(
        image_size=(640,360),
        mode="FULL_COURT",
        image_points=np.array([[0,345],[666,345],[390,170],[257,170]], dtype=np.float32),
        margin_bu=12,
        cells=cells,
        reference_image="reference.png",
    )


def test_far_right_only_selects_far_right_strip():
    cal = _cal([
        _cell(1, 83, 20),
        _cell(2, 84, 50),
        _cell(3, 86, 30),
        _cell(4, 83, 140),
        _cell(5, -1, 20),
    ])
    roi = select_specific_external_roi(cal, "FAR_RIGHT", depth_bu=2)
    assert {c.cell_id for c in roi.cells} == {1, 2}


def test_far_left_only_selects_far_left_strip():
    cal = _cal([
        _cell(1, -1, 20),
        _cell(2, -2, 50),
        _cell(3, -3, 30),
        _cell(4, -1, 140),
    ])
    roi = select_specific_external_roi(cal, "FAR_LEFT", depth_bu=2)
    assert {c.cell_id for c in roi.cells} == {1, 2}


def test_far_baseline_only_selects_first_depth_rows():
    cal = _cal([
        _cell(1, 10, -1),
        _cell(2, 30, -2),
        _cell(3, 40, -3),
        _cell(4, 84, -1),
    ])
    roi = select_specific_external_roi(cal, "FAR_BASELINE", depth_bu=2)
    assert {c.cell_id for c in roi.cells} == {1, 2}


def test_near_right_only_selects_near_half():
    cal = _cal([
        _cell(1, 83, 100),
        _cell(2, 84, 160),
        _cell(3, 83, 40),
    ])
    roi = select_specific_external_roi(cal, "NEAR_RIGHT", depth_bu=2)
    assert {c.cell_id for c in roi.cells} == {1, 2}


def test_near_baseline_selects_beyond_near_baseline():
    cal = _cal([
        _cell(1, 20, 183),
        _cell(2, 40, 184),
        _cell(3, 50, 185),
    ])
    roi = select_specific_external_roi(cal, "NEAR_BASELINE", depth_bu=2)
    assert {c.cell_id for c in roi.cells} == {1, 2}


def test_roi_is_smaller_than_full_external_set():
    cal = _cal([
        _cell(1, 83, 20),
        _cell(2, 84, 20),
        _cell(3, -1, 20),
        _cell(4, 20, -1),
        _cell(5, 20, 183),
    ])
    roi = select_specific_external_roi(cal, "FAR_RIGHT", depth_bu=2)
    assert roi.active_cell_count < roi.full_external_cell_count
    assert roi.reduction_ratio > 0.0


def test_runtime_view_preserves_approved_geometry():
    cal = _cal([_cell(1, 83, 20), _cell(2, -1, 20)])
    roi = select_specific_external_roi(cal, "FAR_RIGHT", depth_bu=2)
    view = calibration_for_specific_roi(cal, roi)
    assert np.array_equal(view.image_points, cal.image_points)
    assert view.reference_image == cal.reference_image
    assert [c.cell_id for c in view.cells] == [1]


@pytest.mark.parametrize("depth", [0, -1])
def test_invalid_depth_rejected(depth):
    cal = _cal([_cell(1, 83, 20)])
    with pytest.raises(ValueError):
        select_specific_external_roi(cal, "FAR_RIGHT", depth_bu=depth)
