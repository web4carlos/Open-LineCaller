from linecaller.dcf import *

def make():
    f=DynamicCourtField(DCFConfig(20,44,5,16,1,1,1,5))
    return f, ImpactEngine(f)

def test_inside_is_silent():
    f,e=make()
    x=e.process_contact(CellIndex(f.x0+5,f.y0+10,0))
    assert not x.is_out
    assert not x.illuminate_floor
    assert x.audio_call is None

def test_boundary_is_silent():
    f,e=make()
    x=e.process_contact(CellIndex(f.x0,f.y0+10,0))
    assert x.region==FieldRegion.BOUNDARY
    assert not x.is_out
    assert x.audio_call is None

def test_external_right_lights_floor_and_calls_out():
    f,e=make()
    x=e.process_contact(CellIndex(f.x1+1,f.y0+10,0))
    assert x.region==FieldRegion.OUT_RIGHT
    assert x.is_out
    assert x.illuminate_floor
    assert x.audio_call=="OUT"

def test_external_far_lights_floor_and_calls_out():
    f,e=make()
    x=e.process_contact(CellIndex(f.x0+5,f.y1+1,0))
    assert x.is_out and x.illuminate_floor and x.audio_call=="OUT"

def test_airborne_cell_is_not_impact():
    f,e=make()
    assert e.process_contact(CellIndex(f.x1+1,f.y0+10,1)) is None

def test_wave_expands_with_uncertainty():
    f,_=make()
    p=CellIndex(f.x0+10,f.y0+20,8)
    low=len(f.illuminate_prediction(p,(1,1,0),uncertainty=0))
    high=len(f.illuminate_prediction(p,(1,1,0),uncertainty=1))
    assert high>low

def test_wave_can_reach_external_field():
    f,_=make()
    p=CellIndex(f.x1-1,f.y0+20,4)
    f.illuminate_prediction(p,(2,0,-1),uncertainty=.2,forward_bias=3)
    assert any(f.region_at_xy(q.x,q.y)==FieldRegion.OUT_RIGHT for q in f.illuminated)
