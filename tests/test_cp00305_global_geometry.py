import cv2
import numpy as np
from linecaller.calibration_center.global_geometry import GlobalPickleballGeometryFitter

def synthetic(color=(0,255,255)):
    f=np.zeros((500,800,3),np.uint8)
    q=np.array([[100,450],[700,450],[540,100],[260,100]],np.float32)
    H=cv2.getPerspectiveTransform(
        np.array([[0,44],[20,44],[20,0],[0,0]],np.float32),q
    )
    def proj(a,b):
        p=np.array([[[*a]],[[*b]]],np.float32)
        z=cv2.perspectiveTransform(p,H)
        return tuple(map(int,z[0,0])),tuple(map(int,z[1,0]))
    for a,b in [
        ((0,0),(20,0)),((0,44),(20,44)),((0,0),(0,44)),((20,0),(20,44)),
        ((0,15),(20,15)),((0,29),(20,29)),
        ((10,0),(10,15)),((10,29),(10,44)),
    ]:
        x,y=proj(a,b); cv2.line(f,x,y,color,5)
    return f

def test_global_fitter_color_agnostic():
    frame=synthetic((255,70,20))
    initial=((100,438),(700,438),(540,112),(260,112))
    r=GlobalPickleballGeometryFitter(search_px=18,step_px=6,min_gain=0).fit(frame,initial)
    assert r.candidate_count > 1
    assert len(r.image_points)==4

def test_global_fitter_multicolor():
    frame=synthetic((0,255,255))
    initial=((100,438),(700,438),(540,112),(260,112))
    r=GlobalPickleballGeometryFitter(search_px=18,step_px=6,min_gain=0).fit(frame,initial)
    assert len(r.image_points)==4

def test_invalid_points_rejected():
    frame=np.zeros((100,100,3),np.uint8)
    r=GlobalPickleballGeometryFitter().fit(frame,[(1,2)])
    assert r.accepted is False
