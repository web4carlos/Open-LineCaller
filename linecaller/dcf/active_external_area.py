from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from linecaller.dcf.external_grid_pillow_watcher import (
    ExternalCell,
    ExternalGridCalibration,
)

ActiveSide = Literal["NEAR", "FAR"]


@dataclass(frozen=True)
class ActiveExternalArea:
    """
    Runtime selection over an already-calibrated external grid.

    IMPORTANT:
    - Calibration owns/builds/projects the complete external grid.
    - Game runtime does NOT rebuild geometry.
    - A flight/rally-side state selects only the external cells that can matter
      for the receiving side.
    - Frame-by-frame Pillow scanning runs only over this selected subset.
    """
    side: ActiveSide
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


def _court_cell_shape(calibration: ExternalGridCalibration) -> tuple[int, int]:
    x_cells = 83
    mode = str(calibration.mode).upper()
    if "HALF" in mode:
        # The existing calibration contract uses ~half the 182.6 BU court length.
        y_cells = 91
    else:
        y_cells = 183
    return x_cells, y_cells


def select_active_external_area(
    calibration: ExternalGridCalibration,
    side: ActiveSide,
) -> ActiveExternalArea:
    """
    Select the U-shaped OUT zone around ONE receiving side.

    FULL COURT:
      FAR  = far baseline band + left/right sideline bands adjacent to FAR half.
      NEAR = near baseline band + left/right sideline bands adjacent to NEAR half.

    HALF COURT:
      The half calibration already represents one receiving half. The selector
      keeps the baseline band plus both side bands and deliberately excludes the
      net-facing edge from being treated as OUT. For ambiguous generic
      HALF_COURT profiles, the requested side identifies which edge is baseline.

    No ball tracking is involved.
    """
    s = str(side).upper()
    if s not in {"NEAR", "FAR"}:
        raise ValueError("side must be NEAR or FAR")

    x_cells, y_cells = _court_cell_shape(calibration)
    mode = str(calibration.mode).upper()
    selected: list[ExternalCell] = []

    if "HALF" in mode:
        # A half-court profile spans baseline<->net. Which image/world edge is
        # the baseline is determined by the half side.
        for c in calibration.cells:
            side_band = (c.gx < 0 or c.gx >= x_cells) and (0 <= c.gy < y_cells)
            if s == "FAR":
                baseline_band = c.gy < 0
            else:
                baseline_band = c.gy >= y_cells
            if side_band or baseline_band:
                selected.append(c)
    else:
        split = y_cells // 2
        for c in calibration.cells:
            if s == "FAR":
                baseline_band = c.gy < 0
                side_band = (c.gx < 0 or c.gx >= x_cells) and (0 <= c.gy < split)
            else:
                baseline_band = c.gy >= y_cells
                side_band = (c.gx < 0 or c.gx >= x_cells) and (split <= c.gy < y_cells)
            if baseline_band or side_band:
                selected.append(c)

    if not selected:
        raise RuntimeError(f"No active external cells selected for side={s}")

    return ActiveExternalArea(
        side=s,  # type: ignore[arg-type]
        cells=tuple(selected),
        full_external_cell_count=len(calibration.cells),
    )


def calibration_for_active_area(
    calibration: ExternalGridCalibration,
    area: ActiveExternalArea,
) -> ExternalGridCalibration:
    """
    Create a runtime VIEW of the saved calibration containing only active cells.
    It does not calculate or alter any geometry.
    """
    return ExternalGridCalibration(
        image_size=calibration.image_size,
        mode=calibration.mode,
        image_points=calibration.image_points.copy(),
        margin_bu=calibration.margin_bu,
        cells=list(area.cells),
        reference_image=calibration.reference_image,
    )
