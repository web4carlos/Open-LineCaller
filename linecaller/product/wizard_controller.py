from __future__ import annotations

from .wizard_models import (
    CalibrationMode,
    MatchWizardState,
    WizardStep,
)


class MatchWizardController:
    def __init__(self):
        self.state = MatchWizardState()

    def select_camera(self, camera_index: int):
        self.state.selected_camera = int(camera_index)

    def continue_from_camera(self):
        self.state.step = WizardStep.CALIBRATION

    def set_calibration_mode(self, mode: CalibrationMode | str):
        self.state.calibration_mode = CalibrationMode(mode)

    def calibration_success(self):
        self.state.calibration_valid = True
        self.state.step = WizardStep.HEALTH

    def calibration_failed(self):
        self.state.calibration_valid = False

    def health_result(self, passed: bool):
        self.state.health_passed = bool(passed)

        if passed and self.state.calibration_valid:
            self.state.step = WizardStep.READY
        else:
            self.state.step = WizardStep.HEALTH

    @property
    def can_start_live(self) -> bool:
        return (
            self.state.step == WizardStep.READY
            and self.state.calibration_valid
            and self.state.health_passed
        )

    def reset(self):
        self.state = MatchWizardState()
