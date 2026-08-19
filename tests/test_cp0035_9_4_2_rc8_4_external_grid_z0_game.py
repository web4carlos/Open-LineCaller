from pathlib import Path
import numpy as np
from PIL import Image
from linecaller.dcf.external_grid_pillow_watcher import BallColorProfile, ExternalCell, ExternalGridCalibration
from linecaller.dcf.external_grid_z0_game import ExternalGridZ0Game, build_runtime_meta, select_side_cells

def prof():
    return BallColorProfile(42.0, 240.0, 245.0, 25.0, 100.0, 100.0)

def cell(cid, gx, gy, poly):
    return ExternalCell(cid, gx, gy, tuple(poly), 400)

def cal(tmp, cells):
    ref = np.zeros((120,160,3), dtype=np.uint8)
    ref[:] = (25,25,25)
    p = tmp / 'ref.png'
    Image.fromarray(ref, 'RGB').save(p)
    return ExternalGridCalibration(image_size=(160,120), mode='FULL_COURT', image_points=np.array([[0,110],[170,110],[110,20],[50,20]], dtype=np.float32), margin_bu=4, cells=cells, reference_image=str(p))

def yellow():
    return np.array([250,245,20], dtype=np.uint8)

def test_active_side(tmp_path):
    c = cal(tmp_path, [cell(1,40,-1,((60,20),(70,20),(70,30),(60,30))), cell(2,40,183,((60,90),(70,90),(70,100),(60,100)))])
    assert [x.cell_id for x in select_side_cells(c, 'FAR', depth_bu=4)] == [1]
    assert [x.cell_id for x in select_side_cells(c, 'NEAR', depth_bu=4)] == [2]

def test_meta(tmp_path):
    m = build_runtime_meta([cell(1,83,20,((80,40),(90,40),(90,50),(80,50)))])[0]
    assert m.floor_position_bu == (83.5,20.5,0.0)
    assert m.expected_scale_px > 0

def test_candidate_then_up_confirms_original_cell(tmp_path):
    c = cal(tmp_path, [cell(1,83,20,((40,70),(60,70),(60,90),(40,90)))])
    g = ExternalGridZ0Game(c, prof(), active_side='FAR', depth_bu=4, min_ball_pixels=2, size_ratio_min=0.1, size_ratio_max=5.0, up_min_bu=0.4, up_max_bu=2.0)
    f = np.asarray(Image.open(c.reference_image).convert('RGB')).copy()
    f[76:82,46:52] = yellow()
    cand = g.scan_for_one_candidate(10, f)
    assert cand is not None
    assert g.check_above_after_fact(10, f) is None
    a = np.asarray(Image.open(c.reference_image).convert('RGB')).copy()
    a[56:62,46:52] = yellow()
    conf = g.check_above_after_fact(11, a)
    assert conf is not None
    assert conf.cell.cell_id == cand.cell.cell_id
    assert conf.candidate_frame == 10

def test_no_background_candidate(tmp_path):
    c = cal(tmp_path, [cell(1,83,20,((40,70),(60,70),(60,90),(40,90)))])
    g = ExternalGridZ0Game(c, prof(), active_side='FAR', depth_bu=4)
    f = np.asarray(Image.open(c.reference_image).convert('RGB')).copy()
    assert g.scan_for_one_candidate(10, f) is None

def test_no_path_down_tracker_xyz_state(tmp_path):
    c = cal(tmp_path, [cell(1,83,20,((40,70),(60,70),(60,90),(40,90)))])
    g = ExternalGridZ0Game(c, prof(), active_side='FAR', depth_bu=4)
    for name in ('down','trajectory','path','history','tracker','xyz'):
        assert not hasattr(g, name)
