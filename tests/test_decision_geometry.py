import pytest
from linecaller.decision.geometry import CourtBoundaryGeometry


def test_inside_signed_distance_positive():
    g = CourtBoundaryGeometry()
    d = g.signed_distance(1.0, 2.0)
    assert d.inside is True
    assert d.signed_distance_m > 0


def test_outside_signed_distance_negative():
    g = CourtBoundaryGeometry()
    d = g.signed_distance(-0.2, 2.0)
    assert d.inside is False
    assert d.nearest_line == "left_sideline"
    assert d.signed_distance_m == pytest.approx(-0.2)


def test_nearest_inside_line():
    g = CourtBoundaryGeometry()
    d = g.signed_distance(0.05, 5.0)
    assert d.nearest_line == "left_sideline"
    assert d.signed_distance_m == pytest.approx(0.05)
