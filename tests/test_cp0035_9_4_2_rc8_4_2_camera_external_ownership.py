from __future__ import annotations

from linecaller.dcf.camera_external_ownership import (
    CameraExternalOwnership,
    calibration_for_camera_owned_area,
    select_camera_owned_external_area,
    zones_for_cell,
)
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridConfig,
)


def _cell(cid, ix, iy, rect, region="OUT_CORNER"):
    return ExternalGridCell(
        cell_id=cid,
        ix=ix,
        iy=iy,
        region=region,
        top_view_rect_bu=rect,
        polygon_image=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        bbox_image=(0, 0, 2, 2),
        expected_floor_ball_diameter_px=4.0,
    )


def _cal(cells):
    return ExternalGridCalibration(
        version=2,
        coverage="FULL_COURT",
        image_size=(640, 360),
        image_points=((0.0, 350.0), (620.0, 350.0), (390.0, 40.0), (230.0, 40.0)),
        config=ExternalGridConfig(
            court_x_bu=83.0,
            full_court_y_bu=182.0,
            half_court_y_bu=91.0,
            cell_bu=1.0,
            margin_bu=10.0,
        ),
        background_image="reference.png",
        image_up_unit=(0.0, -1.0),
        cells=tuple(cells),
    )


def test_camera_receiving_side_intersection_prevents_cross_half_scan():
    cal = _cal([
        _cell(1, 83, 20, (83.0, 20.0, 84.0, 21.0), "OUT_RIGHT"),
        _cell(2, 83, 150, (83.0, 150.0, 84.0, 151.0), "OUT_RIGHT"),
    ])
    ownership = CameraExternalOwnership(
        camera_id="net-right-01",
        mount_position="NET_RIGHT",
        zones=("FAR_RIGHT", "NEAR_RIGHT"),
        depth_bu=3,
    )

    far = select_camera_owned_external_area(cal, ownership, "FAR")
    near = select_camera_owned_external_area(cal, ownership, "NEAR")

    assert [c.cell_id for c in far.cells] == [1]
    assert [c.cell_id for c in near.cells] == [2]


def test_left_camera_never_scans_right_owned_cells():
    cal = _cal([
        _cell(1, -1, 30, (-1.0, 30.0, 0.0, 31.0), "OUT_LEFT"),
        _cell(2, 83, 30, (83.0, 30.0, 84.0, 31.0), "OUT_RIGHT"),
    ])
    ownership = CameraExternalOwnership(
        camera_id="net-left-01",
        mount_position="NET_LEFT",
        zones=("FAR_LEFT",),
        depth_bu=4,
    )

    area = select_camera_owned_external_area(cal, ownership, "FAR")
    assert [c.cell_id for c in area.cells] == [1]


def test_camera_without_current_side_ownership_scans_zero_cells_fail_closed():
    cal = _cal([
        _cell(1, -1, 30, (-1.0, 30.0, 0.0, 31.0), "OUT_LEFT"),
        _cell(2, -1, 150, (-1.0, 150.0, 0.0, 151.0), "OUT_LEFT"),
    ])
    ownership = CameraExternalOwnership(
        camera_id="far-only",
        mount_position="NET_LEFT",
        zones=("FAR_LEFT",),
        depth_bu=4,
    )

    area = select_camera_owned_external_area(cal, ownership, "NEAR")
    assert area.active_zones == ()
    assert area.cells == ()
    assert area.active_cell_count == 0


def test_far_left_corner_belongs_to_far_left_and_far_baseline():
    corner = _cell(
        1,
        -1,
        -1,
        (-1.0, -1.0, 0.0, 0.0),
        "OUT_CORNER",
    )
    cal = _cal([corner])

    memberships = zones_for_cell(cal, corner, depth_bu=3)
    assert "FAR_LEFT" in memberships
    assert "FAR_BASELINE" in memberships


def test_corner_can_be_owned_by_either_adjacent_camera_zone_without_duplicates():
    corner = _cell(
        1,
        83,
        -1,
        (83.0, -1.0, 84.0, 0.0),
        "OUT_CORNER",
    )
    cal = _cal([corner])

    right_owner = CameraExternalOwnership(
        camera_id="right",
        mount_position="NET_RIGHT",
        zones=("FAR_RIGHT", "FAR_BASELINE"),
        depth_bu=3,
    )
    area = select_camera_owned_external_area(cal, right_owner, "FAR")

    assert [c.cell_id for c in area.cells] == [1]


def test_depth_limits_outward_scan_band():
    cal = _cal([
        _cell(1, -1, 40, (-1.0, 40.0, 0.0, 41.0), "OUT_LEFT"),
        _cell(2, -7, 40, (-7.0, 40.0, -6.0, 41.0), "OUT_LEFT"),
    ])
    ownership = CameraExternalOwnership(
        camera_id="left-shallow",
        mount_position="NET_LEFT",
        zones=("FAR_LEFT",),
        depth_bu=3,
    )

    area = select_camera_owned_external_area(cal, ownership, "FAR")
    assert [c.cell_id for c in area.cells] == [1]


def test_runtime_calibration_view_preserves_geometry_and_background():
    cell = _cell(1, 83, 20, (83.0, 20.0, 84.0, 21.0), "OUT_RIGHT")
    cal = _cal([cell])
    ownership = CameraExternalOwnership(
        camera_id="net-right",
        mount_position="NET_RIGHT",
        zones=("FAR_RIGHT",),
        depth_bu=4,
    )
    area = select_camera_owned_external_area(cal, ownership, "FAR")
    view = calibration_for_camera_owned_area(cal, area)

    assert view.image_points == cal.image_points
    assert view.background_image == cal.background_image
    assert view.image_up_unit == cal.image_up_unit
    assert view.cells == (cell,)


def test_duplicate_zone_configuration_is_normalized():
    ownership = CameraExternalOwnership(
        camera_id="cam",
        mount_position="NET_CENTER",
        zones=("FAR_BASELINE", "FAR_BASELINE"),
        depth_bu=4,
    )
    assert ownership.zones == ("FAR_BASELINE",)