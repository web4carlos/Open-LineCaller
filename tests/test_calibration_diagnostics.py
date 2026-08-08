import cv2
import numpy as np

from linecaller.calibration.diagnostics import reprojection_diagnostics


def test_reprojection_diagnostics_zero_for_exact_mapping():
    image = [(0,0),(100,0),(100,100),(0,100)]
    court = [(0,0),(1,0),(1,1),(0,1)]

    h, _ = cv2.findHomography(
        np.asarray(image, dtype=float),
        np.asarray(court, dtype=float),
    )

    diagnostics = reprojection_diagnostics(image, court, h)

    assert len(diagnostics) == 4
    assert max(d.error_px for d in diagnostics) < 1e-6
