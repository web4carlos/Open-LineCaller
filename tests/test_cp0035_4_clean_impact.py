import numpy as np
from tools.dcf_clean_impact_demo import local_ball_px, edge_point, outward_unit

PTS=np.array([[0.,100.],[200.,100.],[140.,20.],[60.,20.]],dtype=np.float32)

def test_ball_cell_shrinks_with_perspective():
    assert local_ball_px(PTS,"right",0.0) > local_ball_px(PTS,"right",1.0)

def test_edge_point_right_near():
    assert np.allclose(edge_point(PTS,"right",0.0),PTS[1])

def test_outward_points_away_from_center():
    p=edge_point(PTS,"right",.5)
    v=outward_unit(PTS,p)
    assert v[0] > 0
