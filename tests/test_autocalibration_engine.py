import cv2
import numpy as np

from linecaller.autocalibration.engine import AutoCalibrationEngine
from linecaller.autocalibration.models import AutoCalibrationDisposition


def test_engine_returns_proposal_object():
    img = np.zeros((600,800,3), dtype=np.uint8)

    cv2.line(img, (100,500), (700,500), (255,255,255), 4)
    cv2.line(img, (200,100), (600,100), (255,255,255), 4)
    cv2.line(img, (100,500), (200,100), (255,255,255), 4)
    cv2.line(img, (700,500), (600,100), (255,255,255), 4)

    proposal = AutoCalibrationEngine().propose(img)

    assert proposal.disposition in set(AutoCalibrationDisposition)
