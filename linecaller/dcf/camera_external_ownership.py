from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibration,
    ExternalGridCell,
)


ReceivingSide = Literal["FAR", "NEAR"]
MountPosition = Literal["NET_LEFT", "NET_RIGHT", "NET_CENTER", "OTHER"]
ExternalZone = Literal[
    "FAR_LEFT",
    "FAR_BASELINE",
    "FAR_RIGHT",
    "NEAR_LEFT",
    "NEAR_BASELINE",
    "NEAR_RIGHT",
]

_VALID_ZONES: frozenset[str] = frozenset(
    {
        "FAR_LEFT",
        "FAR_BASELINE",
        "FAR_RIGHT",
        "NEAR_LEFT",
        "NEAR_BASELINE",
        "NEAR_RIGHT",
    }
)
_VALID_MOUNTS: frozenset[str] = frozenset(
    {"NET_LEFT", "NET_RIGHT", "NET_CENTER", "OTHER"}
)


@dataclass(frozen=True)
class CameraExternalOwnership:
    """
    Physical-camera ownership of the OFFICIAL exterior floor grid.

    Geometry still belongs to calibration. This object only says which
    already-calibrated OUT zones one camera is allowed to inspect.

    Runtime is fail-closed:
      - a FAR rally activates only FAR_* zones owned by this camera;
      - a NEAR rally activates only NEAR_* zones owned by this camera;
      - no matching ownership means zero scanned cells, never full-grid fallback.
    """

    camera_id: str
    mount_position: MountPosition
    zones: tuple[ExternalZone, ...]
    depth_bu: float = 6.0

    def __post_init__(self) -> None:
        cid = str(self.camera_id).strip()
        if not cid:
            raise ValueError("camera_id must not be empty")

        mount = str(self.mount_position).upper()
        if mount not in _VALID_MOUNTS:
            raise ValueError(f"Unsupported mount_position: {self.mount_position!r}")

        if float(self.depth_bu) <= 0.0:
            raise ValueError("depth_bu must be > 0")

        normalized = tuple(str(z).upper() for z in self.zones)
        bad = [z for z in normalized if z not in _VALID_ZONES]
        if bad:
            raise ValueError(f"Unsupported external zone(s): {bad}")

        # Freeze normalized values and remove duplicate zones while preserving order.
        unique = tuple(dict.fromkeys(normalized))
        object.__setattr__(self, "camera_id", cid)
        object.__setattr__(self, "mount_position", mount)
        object.__setattr__(self, "zones", unique)

    def zones_for_receiving_side(self, side: ReceivingSide) -> tuple[ExternalZone, ...]:
        s = _normalize_receiving_side(side)
        prefix = f"{s}_"
        return tuple(z for z in self.zones if z.startswith(prefix))  # type: ignore[return-value]


@dataclass(frozen=True)
class OwnedExternalArea:
    camera_id: str
    receiving_side: ReceivingSide
    mount_position: MountPosition
    active_zones: tuple[ExternalZone, ...]
    cells: tuple[ExternalGridCell, ...]
    full_external_cell_count: int

    @property
    def active_cell_count(self) -> int:
        return len(self.cells)

    @property
    def reduction_ratio(self) -> float:
        if self.full_external_cell_count <= 0:
            return 0.0
        return 1.0 - (self.active_cell_count / self.full_external_cell_count)


def _normalize_receiving_side(side: ReceivingSide | str) -> ReceivingSide:
    s = str(side).strip().upper()
    if s not in {"FAR", "NEAR"}:
        raise ValueError("receiving_side must be FAR or NEAR")
    return s  # type: ignore[return-value]


def _cell_center_bu(cell: ExternalGridCell) -> tuple[float, float]:
    x0, y0, x1, y1 = cell.top_view_rect_bu
    return (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1)))


def zones_for_cell(
    calibration: ExternalGridCalibration,
    cell: ExternalGridCell,
    *,
    depth_bu: float,
) -> frozenset[ExternalZone]:
    """
    Return every physical OUT zone that owns this exterior cell.

    Corner cells deliberately belong to BOTH adjacent zones. Example:
    a cell beyond the FAR baseline and LEFT sideline belongs to
    FAR_BASELINE and FAR_LEFT. This removes corner blind wedges.
    """
    depth = float(depth_bu)
    if depth <= 0.0:
        raise ValueError("depth_bu must be > 0")

    coverage = CalibrationCoverage.parse(calibration.coverage)
    court_x = float(calibration.config.court_x_bu)
    court_y = float(calibration.config.court_y_bu(coverage))
    split_y = 0.5 * court_y
    cx, cy = _cell_center_bu(cell)

    left = cx < 0.0
    right = cx > court_x
    far = cy < 0.0
    near = cy > court_y

    zones: set[str] = set()

    # Baseline strips. Corners remain eligible here too.
    if far and (-cy) <= depth:
        zones.add("FAR_BASELINE")
    if near and (cy - court_y) <= depth:
        zones.add("NEAR_BASELINE")

    # Sideline strips. FAR/NEAR corner cells naturally fall into the
    # corresponding half because cy is beyond the relevant baseline.
    if left and (-cx) <= depth:
        if cy < split_y:
            zones.add("FAR_LEFT")
        else:
            zones.add("NEAR_LEFT")

    if right and (cx - court_x) <= depth:
        if cy < split_y:
            zones.add("FAR_RIGHT")
        else:
            zones.add("NEAR_RIGHT")

    return frozenset(zones)  # type: ignore[return-value]


def select_camera_owned_external_area(
    calibration: ExternalGridCalibration,
    ownership: CameraExternalOwnership,
    receiving_side: ReceivingSide | str,
) -> OwnedExternalArea:
    """
    Intersect:
        calibration exterior cells
        âˆ© camera-owned physical zones
        âˆ© current receiving side.

    No interior geometry can enter because the source calibration is the
    OFFICIAL External Grid and this function only filters that immutable set.
    """
    side = _normalize_receiving_side(receiving_side)
    active_zones = ownership.zones_for_receiving_side(side)

    # Fail closed. A camera without ownership on this receiving side scans none.
    if not active_zones:
        selected: tuple[ExternalGridCell, ...] = ()
    else:
        allowed = set(active_zones)
        chosen: list[ExternalGridCell] = []
        seen: set[int] = set()
        for cell in calibration.cells:
            memberships = zones_for_cell(
                calibration,
                cell,
                depth_bu=ownership.depth_bu,
            )
            if memberships.isdisjoint(allowed):
                continue
            if cell.cell_id in seen:
                continue
            seen.add(cell.cell_id)
            chosen.append(cell)
        selected = tuple(chosen)

    return OwnedExternalArea(
        camera_id=ownership.camera_id,
        receiving_side=side,
        mount_position=ownership.mount_position,
        active_zones=active_zones,
        cells=selected,
        full_external_cell_count=len(calibration.cells),
    )


def calibration_for_camera_owned_area(
    calibration: ExternalGridCalibration,
    area: OwnedExternalArea,
) -> ExternalGridCalibration:
    """
    Runtime VIEW of the calibration restricted to this camera's cells.

    Homography, reference/background, scale and cell polygons are preserved.
    The game loop can consume this view without knowing about camera routing.
    """
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