from tools.dcf_dynamic_volume_demo import height_px

def test_near_volume_is_visually_taller():
    assert height_px(.1,3) > height_px(.9,3)

def test_far_height_remains_positive_before_far_end():
    assert height_px(.9,2) > 0

def test_zero_z_has_zero_visual_height():
    assert height_px(.5,0) == 0
