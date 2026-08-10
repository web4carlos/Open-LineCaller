from __future__ import annotations

from linecaller.calibration.profile import (
    CalibrationProfile,
)

from .artificial_vision_perception import (
    RealArtificialVisionPerceptionAdapter,
)
from .realtime_adapter import (
    RealTimeOfficiatingAdapter,
)


def _load_calibration(
    calibration=None,
    calibration_path=None,
):
    if (
        calibration is not None
        and calibration_path is not None
    ):
        raise ValueError(
            "Provide calibration OR calibration_path, not both."
        )

    if calibration is not None:
        return calibration

    if calibration_path is not None:
        return CalibrationProfile.load(
            calibration_path
        )

    return None


def create_live_pipeline_adapter(
    *,
    detector=None,
    tracker=None,
    bounce_engine=None,
    decision_engine=None,
    calibration=None,
    calibration_path=None,
    minimum_call_confidence: float = 0.98,
):
    """
    Stable factory used by BOTH the live product and validation tools.

    The frame source may be a camera or a recorded file. Perception is shared.

    Without calibration, real bounce events are returned as REVIEW so tools can
    surface them without inventing an IN/OUT call.
    """
    calibration = _load_calibration(
        calibration=calibration,
        calibration_path=calibration_path,
    )

    perception = (
        RealArtificialVisionPerceptionAdapter(
            detector=detector,
            tracker=tracker,
            bounce_engine=bounce_engine,
            decision_engine=decision_engine,
            calibration=calibration,
        )
    )

    return RealTimeOfficiatingAdapter(
        perception,
        minimum_call_confidence=float(
            minimum_call_confidence
        ),
    )
