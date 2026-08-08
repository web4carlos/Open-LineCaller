from pathlib import Path
import numpy as np

from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus


def test_profile_round_trip(tmp_path: Path):
    profile = CalibrationProfile.create(
        name="court_a",
        image_size=(1920, 1080),
        image_points=[(0.0,0.0)] * 6,
        court_points=[(0.0,0.0)] * 6,
        homography=np.eye(3),
        status=CalibrationStatus.VALID,
        mean_error_px=1.0,
        max_error_px=2.0,
        rms_error_px=1.2,
    )
    path = tmp_path / "calibration.json"
    profile.save(path)

    loaded = CalibrationProfile.load(path)

    assert loaded.name == "court_a"
    assert loaded.status == CalibrationStatus.VALID
    assert np.allclose(loaded.homography, np.eye(3))
