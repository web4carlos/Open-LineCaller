from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HealthStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class HealthCheckItem:
    name: str
    status: HealthStatus
    message: str
    recovery: str = ""
    required: bool = True


@dataclass(frozen=True)
class HealthReport:
    items: tuple[HealthCheckItem, ...]

    @property
    def ready(self) -> bool:
        for item in self.items:
            if item.required and item.status != HealthStatus.PASS:
                return False
        return True

    def by_name(self, name: str) -> HealthCheckItem:
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(name)
