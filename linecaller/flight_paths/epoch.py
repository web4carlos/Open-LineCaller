from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import LateralSide, Point3D, Velocity3D
from .path_field import CourtPathField, PathSelection


class FlightEpochTrigger(str, Enum):
    DIRECTION_CHANGE = "DIRECTION_CHANGE"


@dataclass(frozen=True)
class FlightEpoch:
    epoch_id: int
    trigger: FlightEpochTrigger
    selection: PathSelection

    @property
    def p0(self) -> Point3D:
        return self.selection.path.p0

    @property
    def v0(self) -> Velocity3D:
        return self.selection.path.v0

    @property
    def t0_s(self) -> float:
        return self.selection.path.t0_s

    @property
    def predicted_z0(self) -> Point3D:
        return self.selection.landing

    @property
    def predicted_contact_time_s(self) -> float:
        return self.selection.contact_time_s


class FlightEpochManager:
    """A meaningful above-floor direction change starts a new path epoch.

    Pre-serve floor contact alone does not create an outbound serve epoch.  A
    paddle-driven direction change supplies the new P0, V0 and t0.
    """

    def __init__(self, path_field: CourtPathField | None = None):
        self.path_field = path_field or CourtPathField()
        self._next_id = 1
        self._active: FlightEpoch | None = None

    @property
    def active(self) -> FlightEpoch | None:
        return self._active

    def on_direction_change(
        self,
        p0: Point3D,
        v0: Velocity3D,
        *,
        t0_s: float,
        target_lateral: LateralSide | None = None,
        officiating_margin_bu: float = 0.0,
    ) -> FlightEpoch:
        selection = self.path_field.select_from_state(
            p0,
            v0,
            t0_s=t0_s,
            target_lateral=target_lateral,
            officiating_margin_bu=officiating_margin_bu,
        )
        epoch = FlightEpoch(
            epoch_id=self._next_id,
            trigger=FlightEpochTrigger.DIRECTION_CHANGE,
            selection=selection,
        )
        self._next_id += 1
        self._active = epoch
        return epoch

    def on_floor_contact(self, contact: Point3D) -> FlightEpoch | None:
        if abs(contact.z) > 1e-9:
            raise ValueError("floor contact must be on Z=0")
        closed = self._active
        self._active = None
        return closed
