from linecaller.calibration_center.precision_refinement import EditableCourt

def court():
    return EditableCourt.create([(100,400),(700,400),(550,100),(250,100)])

def test_move_near_baseline_moves_two_near_corners():
    c=court()
    c.move_boundary("NEAR_BASELINE",0,10)
    assert c.points[0][1] == 410
    assert c.points[1][1] == 410
    assert c.points[2][1] == 100

def test_move_corner():
    c=court()
    c.move_corner(0,90,420)
    assert c.points[0] == [90.0,420.0]

def test_reset_returns_to_auto():
    c=court()
    c.move_corner(0,1,2)
    c.reset()
    assert c.points[0] == [100.0,400.0]

def test_valid_geometry():
    assert court().valid() is True
