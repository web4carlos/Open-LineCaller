import numpy as np
from linecaller.dcf.z0_floor_glow_renderer import render_soft_floor_glow

def test_glow_is_filled_and_diffuse():
    f = np.zeros((100,140,3), dtype=np.uint8)
    p = ((50,50),(80,50),(82,75),(48,75))
    o = render_soft_floor_glow(f, p)
    assert o.shape == f.shape
    assert int(o.sum()) > 0
    assert int(o[62,65].sum()) > 0
