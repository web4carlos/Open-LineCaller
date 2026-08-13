from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

class CourtMode(str,Enum):
    STANDARD_MATCH="STANDARD_MATCH"
    OPEN_PLAY="OPEN_PLAY"

class CourtStatus(str,Enum):
    IDLE="IDLE"; READY="READY"; LIVE="LIVE"; REVIEW="REVIEW"; OFFLINE="OFFLINE"

@dataclass(frozen=True)
class CourtState:
    court_id:str
    name:str
    mode:CourtMode=CourtMode.STANDARD_MATCH
    status:CourtStatus=CourtStatus.IDLE
    camera_id:Optional[str]=None
    calibration_id:Optional[str]=None
    active_session_id:Optional[str]=None
    score_display:str=""
    last_call:str=""
    players_on_court:int=0
    waiting_players:int=0

@dataclass(frozen=True)
class FacilityState:
    name:str
    courts:tuple[CourtState,...]

class FacilityManager:
    def __init__(self,name="Open LineCaller Facility"):
        self.name=name; self._courts={}
    @property
    def state(self): return FacilityState(self.name,tuple(self._courts.values()))
    def court_count(self): return len(self._courts)
    def add_court(self,court_id,name,*,mode=CourtMode.STANDARD_MATCH):
        if not court_id: raise ValueError("court_id is required")
        if court_id in self._courts: raise ValueError("court already exists")
        self._courts[court_id]=CourtState(court_id,name,mode)
        return self._courts[court_id]
    def get_court(self,court_id):
        if court_id not in self._courts: raise KeyError(f"unknown court: {court_id}")
        return self._courts[court_id]
    def update_court(self,court_id,**changes):
        if "court_id" in changes: raise ValueError("court_id cannot be changed")
        self._courts[court_id]=replace(self.get_court(court_id),**changes)
        return self._courts[court_id]
    def set_status(self,court_id,status): return self.update_court(court_id,status=status)
    def bind_camera(self,court_id,camera_id): return self.update_court(court_id,camera_id=camera_id)
    def bind_calibration(self,court_id,calibration_id): return self.update_court(court_id,calibration_id=calibration_id)
    def start_session(self,court_id,session_id,*,score_display="0-0-2"):
        if self.get_court(court_id).status==CourtStatus.OFFLINE:
            raise RuntimeError("cannot start a session on an offline court")
        return self.update_court(court_id,active_session_id=session_id,status=CourtStatus.LIVE,score_display=score_display,last_call="")
    def end_session(self,court_id):
        return self.update_court(court_id,active_session_id=None,status=CourtStatus.READY,score_display="",last_call="",players_on_court=0)
    def live_courts(self):
        return tuple(c for c in self._courts.values() if c.status in (CourtStatus.LIVE,CourtStatus.REVIEW))
