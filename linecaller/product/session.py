from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProductSession:
    selected_camera: int = 0
    calibration_profile: str | None = None
    developer_mode: bool = False
    active_match_id: str | None = None

    @property
    def has_calibration(self) -> bool:
        return bool(self.calibration_profile)

    def reset_match(self):
        self.active_match_id = None
