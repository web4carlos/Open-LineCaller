from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WizardStep(str, Enum):
    CAMERA = "CAMERA"
    CALIBRATION = "CALIBRATION"
    HEALTH = "HEALTH"
    READY = "READY"


class CalibrationMode(str, Enum):
    AUTO = "AUTO"
    ASSISTED = "ASSISTED"
    MANUAL = "MANUAL"


@dataclass
class MatchWizardState:
    step: WizardStep = WizardStep.CAMERA
    selected_camera: int = 0
    calibration_mode: CalibrationMode = CalibrationMode.AUTO
    calibration_valid: bool = False
    health_passed: bool = False
