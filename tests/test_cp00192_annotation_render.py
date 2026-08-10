import numpy as np
from linecaller.validation.annotation_render import zoom_around_candidate

def test_zoom_preserves_size():
    frame=np.zeros((100,200,3),dtype=np.uint8)
    assert zoom_around_candidate(frame,x=100,y=50).shape==frame.shape

def test_no_center_returns_same():
    frame=np.zeros((20,20,3),dtype=np.uint8)
    assert zoom_around_candidate(frame) is frame
