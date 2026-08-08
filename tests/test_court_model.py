import pytest

from linecaller.calibration.court_model import PickleballCourtModel


def test_official_dimensions():
    court = PickleballCourtModel()
    assert court.width_m == pytest.approx(6.096)
    assert court.length_m == pytest.approx(13.4112)
    assert court.nvz_depth_m == pytest.approx(2.1336)


def test_reference_points_include_nvz_and_corners():
    refs = PickleballCourtModel().reference_points()
    assert "near_left_corner" in refs
    assert "far_right_corner" in refs
    assert "near_nvz_left" in refs
    assert "far_nvz_right" in refs
