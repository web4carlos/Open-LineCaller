from linecaller.court_geometry import CourtCalibration, CourtGeometry


def calibration():
    return CourtCalibration(
        image_points=(
            (0.0, 44.0),
            (20.0, 44.0),
            (20.0, 0.0),
            (0.0, 0.0),
        )
    )


def test_identity_like_mapping_center():
    g = CourtGeometry(calibration())
    x, y = g.image_to_court(10, 22)
    assert abs(x - 10) < 0.01
    assert abs(y - 22) < 0.01


def test_inside_point():
    g = CourtGeometry(calibration())
    r = g.classify(10, 22)
    assert r.inside_court is True
    assert r.geometry_state == "INSIDE"


def test_outside_point():
    g = CourtGeometry(calibration())
    r = g.classify(-2, 22)
    assert r.inside_court is False
    assert r.geometry_state == "OUTSIDE"
    assert r.nearest_line == "LEFT_SIDELINE"


def test_near_line_point():
    g = CourtGeometry(
        calibration(),
        near_line_threshold_in=3,
    )
    r = g.classify(0.10, 22)
    assert r.geometry_state == "NEAR_LINE"
    assert r.nearest_line == "LEFT_SIDELINE"
