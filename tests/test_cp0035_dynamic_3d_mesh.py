import numpy as np
from linecaller.lattice import PerspectiveCourtMesh,MeshConfig
PTS=[[0,1000],[1900,1000],[1150,500],[800,500]]
def m(): return PerspectiveCourtMesh(PTS,MeshConfig(20,44,12,600,1.2))
def test_near_height(): assert m().height_px(0)==600
def test_far_height(): assert m().height_px(1)==0
def test_height_decreases(): assert m().height_px(.2)>m().height_px(.5)>m().height_px(.9)
def test_near_left(): assert np.allclose(m().floor_point(0,0),PTS[0])
def test_far_right(): assert np.allclose(m().floor_point(1,1),PTS[2])
def test_top_above_floor():
    x=m(); assert x.voxel_point(.5,.25,1)[1] < x.voxel_point(.5,.25,0)[1]
def test_far_collapses():
    x=m(); assert np.allclose(x.voxel_point(.5,1,1),x.voxel_point(.5,1,0))
def test_prediction():
    x=m(); assert x.candidate_window((5,10,6),(2,3,-1),0)=={(7,13,5)}
