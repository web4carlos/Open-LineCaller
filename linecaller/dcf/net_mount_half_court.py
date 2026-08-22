from __future__ import annotations

from dataclasses import dataclass

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibration,
    ExternalGridCell,
)


@dataclass(frozen=True)
class NetMountHalfCourtArea:
    """One phone, one local half-court, three callable boundaries."""

    camera_id: str
    mount_position: str
    cells: tuple[ExternalGridCell, ...]
    full_external_cell_count: int
    depth_bu: float
    active_zones: tuple[str, ...] = (
        "HALF_LEFT",
        "HALF_BASELINE",
        "HALF_RIGHT",
    )

    @property
    def active_cell_count(self) -> int:
        return len(self.cells)

    @property
    def reduction_ratio(self) -> float:
        if self.full_external_cell_count <= 0:
            return 0.0
        return 1.0 - (
            self.active_cell_count / self.full_external_cell_count
        )


def _center_bu(cell: ExternalGridCell) -> tuple[float, float]:
    x0, y0, x1, y1 = cell.top_view_rect_bu
    return (
        0.5 * (float(x0) + float(x1)),
        0.5 * (float(y0) + float(y1)),
    )


def select_net_mount_half_court_area(
    calibration: ExternalGridCalibration,
    *,
    camera_id: str,
    mount_position: str,
    depth_bu: float,
) -> NetMountHalfCourtArea:
    """Select only the external floor belonging to one net-mounted phone.

    Local HALF_COURT coordinates are:
      x=0..83 BU       left/right sidelines
      y=0 BU           baseline
      y=court_y BU     net edge

    The phone owns LEFT + BASELINE + RIGHT.  Cells materially beyond the
    net edge are excluded because they belong to the other phone.

    This only filters the immutable EXTERNAL grid.  It never constructs or
    traverses an interior decision grid.
    """

    if CalibrationCoverage.parse(
        calibration.coverage
    ) is not CalibrationCoverage.HALF_COURT:
        raise ValueError(
            "net-mount half-court mode requires HALF_COURT calibration"
        )

    depth = float(depth_bu)
    if depth <= 0.0:
        raise ValueError("depth_bu must be > 0")

    cfg = calibration.config
    court_x = float(cfg.court_x_bu)
    court_y = float(
        cfg.court_y_bu(CalibrationCoverage.HALF_COURT)
    )

    chosen: list[ExternalGridCell] = []
    seen: set[int] = set()

    for cell in calibration.cells:
        cx, cy = _center_bu(cell)

        # Other side of the net: belongs to the other phone.
        if cy > court_y:
            continue

        baseline = cy < 0.0 and (-cy) <= depth
        left = (
            cx < 0.0
            and (-cx) <= depth
            and cy >= -depth
        )
        right = (
            cx > court_x
            and (cx - court_x) <= depth
            and cy >= -depth
        )

        if not (baseline or left or right):
            continue
        if int(cell.cell_id) in seen:
            continue
        seen.add(int(cell.cell_id))
        chosen.append(cell)

    return NetMountHalfCourtArea(
        camera_id=str(camera_id),
        mount_position=str(mount_position),
        cells=tuple(chosen),
        full_external_cell_count=len(calibration.cells),
        depth_bu=depth,
    )


def calibration_for_net_mount_half_court(
    calibration: ExternalGridCalibration,
    area: NetMountHalfCourtArea,
) -> ExternalGridCalibration:
    return ExternalGridCalibration(
        version=calibration.version,
        coverage=calibration.coverage,
        image_size=calibration.image_size,
        image_points=calibration.image_points,
        config=calibration.config,
        background_image=calibration.background_image,
        image_up_unit=calibration.image_up_unit,
        cells=area.cells,
    )
