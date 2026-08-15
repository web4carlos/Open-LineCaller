from __future__ import annotations
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Optional

@dataclass(frozen=True)
class LiveCourtState:
    court_id:str
    name:str
    status:str
    mode:str
    score:str
    last_call:str
    confidence:Optional[float]=None
    nearest_line:str=""
    distance_in:Optional[float]=None
    evidence:str=""
    serving_team:str=""
    server_number:Optional[int]=None
    service_court:str=""
    camera_id:str=""
    session_id:str=""

    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class LiveFacilityState:
    facility_name:str
    sequence:int
    courts:tuple[LiveCourtState,...]

    def to_dict(self):
        return {
            "facility_name":self.facility_name,
            "sequence":self.sequence,
            "courts":[c.to_dict() for c in self.courts],
        }

class LiveStateStore:
    """Thread-safe snapshot store for the React live-referee bridge."""
    def __init__(self,facility_name="Open LineCaller Facility"):
        self.facility_name=facility_name
        self._lock=Lock()
        self._sequence=0
        self._courts={}

    def upsert(self,court:LiveCourtState):
        with self._lock:
            self._courts[court.court_id]=court
            self._sequence+=1
            return self._sequence

    def snapshot(self):
        with self._lock:
            return LiveFacilityState(
                self.facility_name,self._sequence,tuple(self._courts.values())
            )

    def court(self,court_id):
        with self._lock:
            return self._courts.get(court_id)
