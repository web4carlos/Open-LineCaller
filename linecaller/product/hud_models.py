from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StatusLevel(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class LatencyQuality(str, Enum):
    EXCELLENT = "EXCELLENT"
    ACCEPTABLE = "ACCEPTABLE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HUDMetric:
    label: str
    value: str
    level: StatusLevel
    detail: str = ""


@dataclass(frozen=True)
class LastCallPresentation:
    call: str = "-"
    detail: str = "LIVE"
    level: StatusLevel = StatusLevel.UNKNOWN
