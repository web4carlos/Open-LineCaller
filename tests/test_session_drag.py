from linecaller.calibration.session import CalibrationSession


def test_move_point():
    s = CalibrationSession()
    s.add_point(10, 20)
    s.move_point(0, 15, 25)
    assert s.image_points[0] == (15.0, 25.0)


def test_nearest_point_index():
    s = CalibrationSession()
    s.add_point(100, 100)
    s.add_point(300, 300)

    assert s.nearest_point_index(104, 103, 10) == 0
    assert s.nearest_point_index(200, 200, 10) is None
