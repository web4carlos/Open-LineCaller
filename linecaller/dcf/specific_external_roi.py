from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from linecaller.dcf.external_grid_pillow_watcher import (
    ExternalCell,
    ExternalGridCalibration,
)

ExternalZone = Literal[
    "FAR_LEFT",
    "FAR_BASELINE",
    "FAR_RIGHT",
    "NEAR_LEFT",
    "NEAR_BASELINE",
    "NEAR_RIGHT",
]


@dataclass(frozen=True)
class SpecificExternalROI:
    zone: ExternalZone
    depth_bu: int
    cells: tuple[ExternalCell, ...]
    full_external_cell_count: int

    @property
    def active_cell_count(self) -> int:
        return len(self.cells)

    @property
    def reduction_ratio(self) -> float:
        if self.full_external_cell_count <= 0:
            return 0.0
        return 1.0 - (self.active_cell_count / self.full_external_cell_count)


def _shape(calibration: ExternalGridCalibration) -> tuple[int, int]:
    x_cells = 83
    mode = str(calibration.mode).upper()
    y_cells = 91 if "HALF" in mode else 183
    return x_cells, y_cells


def select_specific_external_roi(
    calibration: ExternalGridCalibration,
    zone: ExternalZone,
    *,
    depth_bu: int,
) -> SpecificExternalROI:
    z = str(zone).upper()
    valid = {
        "FAR_LEFT", "FAR_BASELINE", "FAR_RIGHT",
        "NEAR_LEFT", "NEAR_BASELINE", "NEAR_RIGHT",
    }
    if z not in valid:
        raise ValueError(f"zone must be one of {sorted(valid)}")

    depth = int(depth_bu)
    if depth <= 0:
        raise ValueError("depth_bu must be > 0")

    x_cells, y_cells = _shape(calibration)
    split = y_cells // 2
    mode = str(calibration.mode).upper()
    selected: list[ExternalCell] = []

    for c in calibration.cells:
        gx, gy = c.gx, c.gy

        if "HALF" in mode:
            if z.endswith("_LEFT"):
                keep = (-depth <= gx < 0) and (0 <= gy < y_cells)
            elif z.endswith("_RIGHT"):
                keep = (x_cells <= gx < x_cells + depth) and (0 <= gy < y_cells)
            elif z == "FAR_BASELINE":
                keep = (-depth <= gy < 0) and (0 <= gx < x_cells)
            elif z == "NEAR_BASELINE":
                keep = (y_cells <= gy < y_cells + depth) and (0 <= gx < x_cells)
            else:
                keep = False
        else:
            if z == "FAR_LEFT":
                keep = (-depth <= gx < 0) and (0 <= gy < split)
            elif z == "FAR_BASELINE":
                keep = (-depth <= gy < 0) and (0 <= gx < x_cells)
            elif z == "FAR_RIGHT":
                keep = (x_cells <= gx < x_cells + depth) and (0 <= gy < split)
            elif z == "NEAR_LEFT":
                keep = (-depth <= gx < 0) and (split <= gy < y_cells)
            elif z == "NEAR_BASELINE":
                keep = (y_cells <= gy < y_cells + depth) and (0 <= gx < x_cells)
            elif z == "NEAR_RIGHT":
                keep = (x_cells <= gx < x_cells + depth) and (split <= gy < y_cells)
            else:
                keep = False

        if keep:
            selected.append(c)

    if not selected:
        raise RuntimeError(f"No visible cells selected for zone={z}, depth_bu={depth}")

    return SpecificExternalROI(
        zone=z,  # type: ignore[arg-type]
        depth_bu=depth,
        cells=tuple(selected),
        full_external_cell_count=len(calibration.cells),
    )


def calibration_for_specific_roi(
    calibration: ExternalGridCalibration,
    roi: SpecificExternalROI,
) -> ExternalGridCalibration:
    return ExternalGridCalibration(
        image_size=calibration.image_size,
        mode=calibration.mode,
        image_points=calibration.image_points.copy(),
        margin_bu=calibration.margin_bu,
        cells=list(roi.cells),
        reference_image=calibration.reference_image,
    )
