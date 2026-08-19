from types import SimpleNamespace

import numpy as np
import pytest

from linecaller.dcf.ball_in_mesh_search import DCFMeshProjector
from linecaller.dcf.bounce_floor_glow import (
    BounceFloorGlowController,
    BounceFloorGlowRenderer,
    BounceFloorProjector,
    BounceGlowConfig,
    BounceGlowEvent,
)
from linecaller.dcf.models import CellIndex


def make_projector():
    return DCFMeshProjector(
        np.asarray(
            [
                [100.0, 450.0],
                [800.0, 450.0],
                [570.0, 110.0],
                [330.0, 110.0],
            ],
            dtype=np.float64,
        )
    )


def fake_step(*, hit_cell, contact_cell=None, floor_trigger=False, generation=1, contact_plane=False):
    hit = SimpleNamespace(
        found=hit_cell is not None,
        index=hit_cell,
        ball_generation=generation,
    )
    state = SimpleNamespace(ball_generation=generation)
    return SimpleNamespace(
        hit=hit,
        state=state,
        contact_plane_reached=contact_plane,
        floor_trigger=floor_trigger,
        predicted_contact_cell=contact_cell,
    )


def test_default_timing_matches_approved_long_visible_envelope():
    cfg = BounceGlowConfig()
    assert cfg.fade_in_ms == 180.0
    assert cfg.hold_ms == 1400.0
    assert cfg.fade_out_ms == 900.0
    assert cfg.duration_ms == 2480.0


def test_floor_projector_accepts_only_z0():
    p = make_projector()
    floor = BounceFloorProjector(p)
    good = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    q = floor.project(good)
    assert q.cell == good
    assert q.nominal_diameter_px > 0.0
    with pytest.raises(ValueError):
        floor.project(CellIndex(good.x, good.y, 1))


def test_visual_footprint_is_soft_disc_not_technical_cell_square():
    p = make_projector()
    floor = BounceFloorProjector(p)
    cell = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    q = floor.project(cell)
    assert len(q.core_polygon) >= 16
    assert len(q.halo_polygon) >= 16
    # More than four vertices is an explicit guard against exposing the cell
    # rectangle as the production visual shape.
    assert q.core_polygon.shape[0] != 4
    assert q.halo_polygon.shape[0] != 4


def test_envelope_fades_in_holds_and_fades_out():
    p = make_projector()
    q = BounceFloorProjector(p).project(CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0))
    event = BounceGlowEvent(q.cell, q, (0, 210, 255), 10.0, 1)
    r = BounceFloorGlowRenderer(BounceGlowConfig(fade_in_ms=100, hold_ms=200, fade_out_ms=300))
    assert r.envelope(event, 10.0) == pytest.approx(0.0)
    assert 0.0 < r.envelope(event, 10.05) < 1.0
    assert r.envelope(event, 10.15) == pytest.approx(1.0)
    assert 0.0 < r.envelope(event, 10.45) < 1.0
    assert r.envelope(event, 10.61) == pytest.approx(0.0)


def test_runtime_color_is_not_hardcoded_decision_mapping():
    p = make_projector()
    cell = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    q = BounceFloorProjector(p).project(cell)
    cfg = BounceGlowConfig(fade_in_ms=0, hold_ms=1000, fade_out_ms=0)
    r = BounceFloorGlowRenderer(cfg)
    frame = np.zeros((500, 900, 3), dtype=np.uint8)
    a = r.render(frame, BounceGlowEvent(cell, q, (255, 0, 0), 0.0, 1), 0.1)
    b = r.render(frame, BounceGlowEvent(cell, q, (0, 0, 255), 0.0, 1), 0.1)
    assert not np.array_equal(a, b)


def test_renderer_changes_local_floor_but_preserves_original_video_elsewhere():
    p = make_projector()
    cell = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    q = BounceFloorProjector(p).project(cell)
    cfg = BounceGlowConfig(fade_in_ms=0, hold_ms=1000, fade_out_ms=0)
    r = BounceFloorGlowRenderer(cfg)
    frame = np.full((500, 900, 3), 120, dtype=np.uint8)
    out = r.render(frame, BounceGlowEvent(cell, q, (0, 210, 255), 0.0, 1), 0.1)
    cx, cy = round(q.center_x), round(q.center_y)
    assert np.any(out[cy, cx] != frame[cy, cx])
    assert np.array_equal(out[5, 5], frame[5, 5])


def test_z1_down_prediction_lights_predicted_z0_not_current_z1():
    p = make_projector()
    c = BounceFloorGlowController(p)
    z1 = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 1)
    z0 = CellIndex(p.field.x0 + 41, p.field.y0 + 82, 0)
    step = fake_step(
        hit_cell=z1,
        contact_cell=z0,
        floor_trigger=True,
        contact_plane=True,
    )
    event = c.consume_wave_step(step, now_s=1.0, color_bgr=(0, 210, 255))
    assert event is not None
    assert event.cell == z0
    assert event.cell != z1


def test_direct_descending_z0_trigger_lights_that_exact_contact_cell():
    p = make_projector()
    c = BounceFloorGlowController(p)
    z0 = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    step = fake_step(
        hit_cell=z0,
        contact_cell=z0,
        floor_trigger=True,
        contact_plane=True,
    )
    event = c.consume_wave_step(step, now_s=1.0, color_bgr=(0, 210, 255))
    assert event is not None
    assert event.cell == z0


def test_contact_plane_flag_alone_never_lights_floor():
    p = make_projector()
    c = BounceFloorGlowController(p)
    z0 = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    step = fake_step(
        hit_cell=z0,
        contact_cell=None,
        floor_trigger=False,
        contact_plane=True,
    )
    assert c.consume_wave_step(step, now_s=1.0, color_bgr=(0, 210, 255)) is None
    assert c.active_events == ()


def test_non_contact_local_hit_never_lights_floor():
    p = make_projector()
    c = BounceFloorGlowController(p)
    air = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 3)
    step = fake_step(hit_cell=air, floor_trigger=False)
    assert c.consume_wave_step(step, now_s=1.0, color_bgr=(0, 210, 255)) is None


def test_ball_generation_change_clears_old_visual_events():
    p = make_projector()
    c = BounceFloorGlowController(p)
    z0 = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    c.trigger_contact(z0, now_s=1.0, color_bgr=(0, 210, 255), ball_generation=1)
    assert len(c.active_events) == 1
    air = CellIndex(z0.x, z0.y, 4)
    c.consume_wave_step(fake_step(hit_cell=air, generation=2), now_s=1.1, color_bgr=(0, 210, 255))
    assert c.active_events == ()


def test_expired_event_is_pruned():
    p = make_projector()
    c = BounceFloorGlowController(
        p,
        config=BounceGlowConfig(fade_in_ms=0, hold_ms=10, fade_out_ms=10),
    )
    z0 = CellIndex(p.field.x0 + 40, p.field.y0 + 80, 0)
    c.trigger_contact(z0, now_s=1.0, color_bgr=(0, 210, 255), ball_generation=1)
    frame = np.zeros((500, 900, 3), dtype=np.uint8)
    c.render(frame, now_s=2.0)
    assert c.active_events == ()


def test_controller_has_no_mesh_or_wave_render_api():
    c = BounceFloorGlowController(make_projector())
    assert not hasattr(c, "draw_mesh")
    assert not hasattr(c, "draw_wave")
