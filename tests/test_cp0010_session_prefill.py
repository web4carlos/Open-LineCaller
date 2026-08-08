import pytest

from linecaller.calibration.session import CalibrationSession


def test_prefill_outer_corners():
    s = CalibrationSession()
    corners = [(1,10),(10,10),(9,1),(2,1)]

    s.prefill_outer_corners(corners)

    assert len(s.image_points) == 4
    assert s.image_points[0] == (1.0,10.0)
    assert s.image_points[3] == (2.0,1.0)


def test_prefill_requires_four():
    s = CalibrationSession()

    with pytest.raises(ValueError):
        s.prefill_outer_corners([(0,0),(1,1)])
