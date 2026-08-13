import numpy as np

from linecaller.bounce.models import BounceEvent, BounceEvidence
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus
from linecaller.decision.engine import DecisionEngine
from linecaller.decision.fusion import CourtFusion
from linecaller.decision.models import Decision


def bounce(x, y, conf=.95):
    return BounceEvent(
        20,
        float(x),
        float(y),
        conf,
        BounceEvidence(6.0, -6.0, 1.0, .95, 12.0),
    )


def profile(status=CalibrationStatus.VALID):
    h = np.array(
        [[.01, 0, 0], [0, .01, 0], [0, 0, 1]],
        dtype=float,
    )

    return CalibrationProfile.create(
        name="p",
        image_size=(1920, 1080),
        image_points=[(0.0, 0.0)] * 6,
        court_points=[(0.0, 0.0)] * 6,
        homography=h,
        status=status,
        mean_error_px=1.0,
        max_error_px=2.0,
        rms_error_px=1.2,
    )


def test_legacy_constructor_and_decide_are_supported():
    engine = DecisionEngine(
        fusion=CourtFusion(
            bounce_localization_error_px=1.0
        ),
        min_bounce_confidence=.70,
    )

    result = engine.decide(
        bounce(100, 200),
        profile(),
    )

    assert result.decision == Decision.IN


def test_new_geometry_api_still_supported():
    engine = DecisionEngine()

    result = engine.decide(
        geometry_state="INSIDE",
        nearest_line="LEFT_SIDELINE",
        signed_distance_ft=1.0,
        bounce_confidence=0.8,
        bounce_score=0.8,
    )

    assert result.call == "IN"
