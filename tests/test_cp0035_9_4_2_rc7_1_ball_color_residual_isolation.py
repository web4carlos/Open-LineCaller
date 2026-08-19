from __future__ import annotations

import numpy as np

from linecaller.dcf.ball_color_residual_isolator import (
    BallColorResidualIsolator,
    LockedResidualColorProfile,
)


def _disk(img, cx, cy, r, rgb):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (xx-cx)**2 + (yy-cy)**2 <= r*r
    img[m] = np.array(rgb, dtype=np.uint8)


def _template(rgb):
    a = np.full((15, 15, 3), 30, dtype=np.uint8)
    _disk(a, 7, 7, 4, rgb)
    return a


def test_same_background_has_no_ball_residual():
    bg = np.full((25, 25, 3), (60, 80, 120), dtype=np.uint8)
    profile = LockedResidualColorProfile.from_template(_template((225, 240, 70)))
    ev = BallColorResidualIsolator(profile).isolate(bg, bg, expected_diameter_px=8.0)
    assert ev.residual_present is False
    assert ev.component_area_px == 0


def test_locked_color_isolated_from_changed_background():
    bg = np.full((25, 25, 3), (60, 80, 120), dtype=np.uint8)
    cur = bg.copy()
    cur[:, :4] = (230, 70, 60)          # unrelated changed strip
    _disk(cur, 12, 12, 4, (225, 240, 70))
    profile = LockedResidualColorProfile.from_template(_template((225, 240, 70)))
    ev = BallColorResidualIsolator(profile).isolate(cur, bg, expected_diameter_px=8.0)
    assert ev.residual_present is True
    assert ev.component_area_px >= 20
    assert ev.centroid_xy is not None
    assert abs(ev.centroid_xy[0]-12) < 1.0
    assert abs(ev.centroid_xy[1]-12) < 1.0


def test_wrong_color_change_is_rejected():
    bg = np.full((25, 25, 3), (60, 80, 120), dtype=np.uint8)
    cur = bg.copy()
    _disk(cur, 12, 12, 4, (235, 45, 45))
    profile = LockedResidualColorProfile.from_template(_template((225, 240, 70)))
    ev = BallColorResidualIsolator(profile).isolate(cur, bg, expected_diameter_px=8.0)
    assert ev.residual_present is False


def test_color_is_not_hardcoded_to_yellow():
    bg = np.full((25, 25, 3), (80, 80, 80), dtype=np.uint8)
    cur = bg.copy()
    orange = (245, 125, 35)
    _disk(cur, 12, 12, 4, orange)
    profile = LockedResidualColorProfile.from_template(_template(orange))
    ev = BallColorResidualIsolator(profile).isolate(cur, bg, expected_diameter_px=8.0)
    assert ev.residual_present is True


def test_output_isolation_keeps_only_selected_component():
    bg = np.full((25, 25, 3), (50, 70, 100), dtype=np.uint8)
    cur = bg.copy()
    ball = (225, 240, 70)
    _disk(cur, 12, 12, 4, ball)
    _disk(cur, 2, 2, 2, ball)            # same color, outside expected local area
    profile = LockedResidualColorProfile.from_template(_template(ball))
    ev = BallColorResidualIsolator(profile).isolate(cur, bg, expected_diameter_px=8.0)
    assert ev.residual_present is True
    assert np.count_nonzero(ev.isolated_rgb[:5, :5]) == 0
    assert np.count_nonzero(ev.isolated_rgb[8:17, 8:17]) > 0
