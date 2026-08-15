from dataclasses import dataclass
from enum import Enum

class FieldRegion(str, Enum):
    INSIDE="INSIDE"
    BOUNDARY="BOUNDARY"
    OUT_LEFT="OUT_LEFT"
    OUT_RIGHT="OUT_RIGHT"
    OUT_NEAR="OUT_NEAR"
    OUT_FAR="OUT_FAR"
    OUT_CORNER="OUT_CORNER"

@dataclass(frozen=True, order=True)
class CellIndex:
    x:int
    y:int
    z:int

@dataclass(frozen=True)
class DCFConfig:
    court_x_cells:int=83
    court_y_cells:int=182
    margin_cells:int=24
    z_cells:int=48
    boundary_cells:int=1
    active_radius_xy:int=2
    active_radius_z:int=2
    max_active_radius:int=8

@dataclass(frozen=True)
class ImpactEvent:
    cell: CellIndex
    region: FieldRegion
    is_out: bool
    illuminate_floor: bool
    audio_call: str | None
