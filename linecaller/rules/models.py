from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RallyPhase(str, Enum):
    READY = "READY"
    SERVE_FLIGHT = "SERVE_FLIGHT"
    RETURN_FLIGHT = "RETURN_FLIGHT"
    THIRD_SHOT_FLIGHT = "THIRD_SHOT_FLIGHT"
    OPEN_RALLY = "OPEN_RALLY"
    REVIEW = "REVIEW"
    ENDED = "ENDED"


class TeamSide(str, Enum):
    A = "A"
    B = "B"
    UNKNOWN = "UNKNOWN"


class RallyEventType(str, Enum):
    START_RALLY = "START_RALLY"
    BOUNCE = "BOUNCE"
    HIT = "HIT"
    FAULT = "FAULT"
    REVIEW = "REVIEW"
    END_RALLY = "END_RALLY"


@dataclass(frozen=True)
class RallyEvent:
    event_type: RallyEventType
    frame: int
    team: TeamSide = TeamSide.UNKNOWN
    decision: str = ""
    zone: str = ""
    reason: str = ""


@dataclass(frozen=True)
class RallyState:
    phase: RallyPhase = RallyPhase.READY
    serving_team: TeamSide = TeamSide.UNKNOWN
    receiving_team: TeamSide = TeamSide.UNKNOWN
    bounce_count: int = 0
    hit_count: int = 0
    rally_active: bool = False
    review_required: bool = False
    last_frame: int = -1


@dataclass(frozen=True)
class RulesResult:
    state: RallyState
    accepted: bool
    ruling: str
    reason: str
