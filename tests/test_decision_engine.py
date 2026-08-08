import numpy as np

from linecaller.bounce.models import BounceEvent, BounceEvidence
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus
from linecaller.decision.engine import DecisionEngine
from linecaller.decision.fusion import CourtFusion
from linecaller.decision.models import Decision


def b(x,y,conf=.95):
    return BounceEvent(
        20, float(x), float(y), conf,
        BounceEvidence(6.0,-6.0,1.0,.95,12.0)
    )


def profile(status=CalibrationStatus.VALID, err=1.0):
    h=np.array([[.01,0,0],[0,.01,0],[0,0,1]], dtype=float)
    return CalibrationProfile.create(
        name="p",
        image_size=(1920,1080),
        image_points=[(0.0,0.0)]*6,
        court_points=[(0.0,0.0)]*6,
        homography=h,
        status=status,
        mean_error_px=err,
        max_error_px=2.0,
        rms_error_px=1.2,
    )


def engine():
    return DecisionEngine(
        fusion=CourtFusion(bounce_localization_error_px=1.0),
        min_bounce_confidence=.70,
    )


def test_clear_inside_is_in():
    # x=1m, y=2m: comfortably inside
    r=engine().decide(b(100,200), profile())
    assert r.decision == Decision.IN


def test_clear_outside_is_out():
    # x=-0.20m -> image x=-20 px
    r=engine().decide(b(-20,200), profile())
    assert r.decision == Decision.OUT


def test_close_boundary_is_review():
    # x=0.01m = 1px. uncertainty sqrt(1^2+1^2)*0.01m ~= 14.1mm
    r=engine().decide(b(1,200), profile())
    assert r.decision == Decision.REVIEW


def test_invalid_calibration_is_review():
    r=engine().decide(
        b(100,200),
        profile(CalibrationStatus.INVALID),
    )
    assert r.decision == Decision.REVIEW


def test_low_bounce_confidence_is_review():
    r=engine().decide(b(100,200,.30), profile())
    assert r.decision == Decision.REVIEW
