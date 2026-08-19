from __future__ import annotations

import numpy as np
from PIL import Image

from linecaller.dcf.cell_residual_ball_verifier import (
    CellResidualBallVerifier,
    CellResidualConfig,
)


def _blank(size=25, value=70):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_no_change_is_not_foreground():
    bg = _blank()
    ev = CellResidualBallVerifier().verify(bg, bg, expected_diameter_px=7.0)
    assert ev.residual_present is False
    assert ev.center_changed_fraction == 0.0
    assert ev.center_mean_difference == 0.0
    assert ev.changed_bbox is None


def test_compact_center_change_is_detected():
    bg = _blank()
    cur = bg.copy()
    yy, xx = np.ogrid[:25, :25]
    mask = (xx - 12) ** 2 + (yy - 12) ** 2 <= 4 ** 2
    cur[mask] = np.array([220, 240, 80], dtype=np.uint8)

    ev = CellResidualBallVerifier().verify(cur, bg, expected_diameter_px=7.0)
    assert ev.residual_present is True
    assert ev.center_changed_fraction > 0.45
    assert ev.center_mean_difference > 28.0
    assert ev.changed_bbox is not None


def test_motion_only_at_edge_does_not_fake_expected_center():
    bg = _blank()
    cur = bg.copy()
    cur[:, :5] = 220

    ev = CellResidualBallVerifier().verify(cur, bg, expected_diameter_px=7.0)
    assert ev.residual_present is False
    assert ev.full_changed_fraction > 0.0
    assert ev.center_changed_fraction < 0.45


def test_accepts_pil_images():
    bg = Image.fromarray(_blank())
    cur_arr = _blank()
    cur_arr[9:16, 9:16] = 230
    cur = Image.fromarray(cur_arr)

    ev = CellResidualBallVerifier().verify(cur, bg, expected_diameter_px=7.0)
    assert ev.residual_present is True


def test_identity_is_not_part_of_residual_contract():
    # The verifier deliberately has no color/name/template identity argument.
    # It only proves local foreground presence; BallInMeshSearcher still owns WHO.
    verifier = CellResidualBallVerifier(CellResidualConfig())
    assert not hasattr(verifier, "ball_identity")
