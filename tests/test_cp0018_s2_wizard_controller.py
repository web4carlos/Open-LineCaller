from linecaller.product.wizard_controller import MatchWizardController
from linecaller.product.wizard_models import (
    CalibrationMode,
    WizardStep,
)


def test_wizard_camera_to_calibration():
    c = MatchWizardController()

    c.select_camera(2)
    c.continue_from_camera()

    assert c.state.selected_camera == 2
    assert c.state.step == WizardStep.CALIBRATION


def test_wizard_calibration_to_health():
    c = MatchWizardController()

    c.continue_from_camera()
    c.set_calibration_mode(CalibrationMode.AUTO)
    c.calibration_success()

    assert c.state.calibration_valid is True
    assert c.state.step == WizardStep.HEALTH


def test_wizard_health_to_ready():
    c = MatchWizardController()

    c.continue_from_camera()
    c.calibration_success()
    c.health_result(True)

    assert c.state.step == WizardStep.READY
    assert c.can_start_live is True


def test_failed_health_cannot_start():
    c = MatchWizardController()

    c.continue_from_camera()
    c.calibration_success()
    c.health_result(False)

    assert c.state.step == WizardStep.HEALTH
    assert c.can_start_live is False
