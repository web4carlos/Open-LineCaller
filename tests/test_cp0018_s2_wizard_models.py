from linecaller.product.wizard_models import (
    CalibrationMode,
    MatchWizardState,
    WizardStep,
)


def test_wizard_defaults():
    s = MatchWizardState()

    assert s.step == WizardStep.CAMERA
    assert s.calibration_mode == CalibrationMode.AUTO
    assert s.calibration_valid is False
    assert s.health_passed is False
