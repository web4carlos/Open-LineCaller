import cv2
import numpy as np
from linecaller.calibration_center import AutoCourtCalibrator


def blue_court():
    f=np.zeros((720,1280,3),dtype=np.uint8)
    p=np.array([[60,670],[1220,670],[840,240],[440,240]],dtype=np.int32)
    cv2.fillConvexPoly(f,p,(180,80,30))
    cv2.polylines(f,[p],True,(255,255,255),8)
    return f


def white_line_court():
    f=np.zeros((720,1280,3),dtype=np.uint8)
    p=np.array([[60,670],[1220,670],[840,240],[440,240]],dtype=np.int32)
    cv2.polylines(f,[p],True,(255,255,255),9)
    cv2.line(f,(250,500),(1030,500),(255,255,255),6)
    return f


def test_hybrid_prefers_surface_when_available():
    r=AutoCourtCalibrator(min_confidence=.10).detect(blue_court())
    assert r.calibration is not None
    assert "SURFACE" in r.reason


def test_hybrid_falls_back_to_lines_without_blue():
    r=AutoCourtCalibrator(min_confidence=.10).detect(white_line_court())
    assert r.calibration is not None
    assert "LINES" in r.reason
