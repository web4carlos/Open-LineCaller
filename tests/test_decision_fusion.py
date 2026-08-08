import numpy as np

from linecaller.bounce.models import BounceEvent, BounceEvidence
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus
from linecaller.decision.fusion import CourtFusion


def bounce(x=100.0, y=100.0, confidence=0.95):
    return BounceEvent(
        frame_number=10,
        x=x,
        y=y,
        confidence=confidence,
        evidence=BounceEvidence(5.0, -5.0, 1.0, 0.95, 10.0),
    )


def profile(status=CalibrationStatus.VALID, mean_error_px=1.0):
    # 100 px == 1 meter
    h = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return CalibrationProfile.create(
        name="synthetic",
        image_size=(1920,1080),
        image_points=[(0.0,0.0)] * 6,
        court_points=[(0.0,0.0)] * 6,
        homography=h,
        status=status,
        mean_error_px=mean_error_px,
        max_error_px=2.0,
        rms_error_px=1.2,
    )


def test_fusion_projects_to_court():
    ctx = CourtFusion(bounce_localization_error_px=1.0).build_context(
        bounce(100,200),
        profile(),
    )
    assert ctx.court_x_m == 1.0
    assert ctx.court_y_m == 2.0
    assert ctx.local_m_per_px > 0


def test_invalid_calibration_blocks_geometry():
    ctx = CourtFusion().build_context(
        bounce(),
        profile(CalibrationStatus.INVALID),
    )
    assert ctx.calibration_valid is False
    assert ctx.court_x_m is None
