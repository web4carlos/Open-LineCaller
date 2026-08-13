import cv2
import numpy as np
from linecaller.calibration_center.refinement import CourtGeometryRefiner

def scene(color):
    f=np.zeros((500,800,3),dtype=np.uint8)
    q=np.array([[80,450],[720,450],[550,120],[250,120]],np.int32)
    cv2.polylines(f,[q],True,color,8)
    return f,q

def test_refiner_does_not_require_white():
    frame,q=scene((0,255,255))
    initial=((92,440),(708,440),(542,130),(258,130))
    r=CourtGeometryRefiner(search_radius_px=28,min_gain=0.0).refine(frame,initial)
    assert len(r.image_points)==4

def test_refiner_handles_blue_lines():
    frame,q=scene((255,80,20))
    initial=((92,440),(708,440),(542,130),(258,130))
    r=CourtGeometryRefiner(search_radius_px=28,min_gain=0.0).refine(frame,initial)
    assert len(r.image_points)==4

def test_bad_input_is_rejected():
    f=np.zeros((100,100,3),dtype=np.uint8)
    r=CourtGeometryRefiner().refine(f,[(0,0)])
    assert r.accepted is False
