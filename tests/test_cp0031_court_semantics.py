from linecaller.court_semantics import CourtLine, CourtSemanticModel, CourtZone

def m(): return CourtSemanticModel(line_tolerance_in=2.0)

def test_far_left_service_zone(): assert m().classify(5,5).zone == CourtZone.FAR_LEFT_SERVICE
def test_far_right_service_zone(): assert m().classify(15,5).zone == CourtZone.FAR_RIGHT_SERVICE
def test_near_left_service_zone(): assert m().classify(5,40).zone == CourtZone.NEAR_LEFT_SERVICE
def test_near_right_service_zone(): assert m().classify(15,40).zone == CourtZone.NEAR_RIGHT_SERVICE
def test_far_kitchen(): assert m().classify(5,18).zone == CourtZone.FAR_KITCHEN
def test_near_kitchen(): assert m().classify(15,26).zone == CourtZone.NEAR_KITCHEN

def test_net_zone():
    r=m().classify(10,22)
    assert r.zone == CourtZone.NET
    assert r.nearest_line == CourtLine.NET

def test_outside_zone():
    r=m().classify(-1,20)
    assert r.zone == CourtZone.OUTSIDE
    assert r.inside_outer_court is False

def test_far_nvz_line_is_semantic_boundary():
    r=m().classify(5,15)
    assert r.zone == CourtZone.FAR_KITCHEN
    assert r.nearest_line == CourtLine.FAR_NVZ_LINE
    assert r.on_line is True

def test_centerline_only_service_segment():
    service=m().classify(10,8)
    kitchen=m().classify(10,18)
    assert service.nearest_line == CourtLine.FAR_CENTERLINE
    assert kitchen.nearest_line != CourtLine.FAR_CENTERLINE
