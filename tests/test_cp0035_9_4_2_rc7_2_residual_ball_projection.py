from __future__ import annotations

import math
import numpy as np

from linecaller.dcf.ball_color_residual_isolator import BallColorResidualEvidence
from linecaller.dcf.residual_ball_projection import ResidualBallProjectionEstimator


def evidence_from_mask(mask: np.ndarray, strength: np.ndarray | None = None):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    raw = np.zeros((h, w, 3), dtype=np.uint8)
    if strength is None:
        strength = np.full((h, w), 180, dtype=np.uint8)
    raw[mask] = np.repeat(strength[..., None], 3, axis=2)[mask]
    isolated = np.zeros_like(raw)
    isolated[mask] = (220, 240, 70)
    ys, xs = np.nonzero(mask)
    if len(xs):
        bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
        centroid = (float(xs.mean()), float(ys.mean()))
    else:
        bbox = None
        centroid = None
    return BallColorResidualEvidence(
        residual_present=bool(mask.any()),
        component_area_px=int(mask.sum()),
        component_bbox=bbox,
        centroid_xy=centroid,
        changed_fraction=float(mask.mean()),
        color_candidate_fraction=float(mask.mean()),
        raw_residual_rgb=raw,
        color_mask_u8=(mask.astype(np.uint8)*255),
        isolated_rgb=isolated,
    )


def test_irregular_projection_does_not_require_circle():
    m = np.zeros((21, 25), dtype=bool)
    m[8:13, 8:17] = True
    m[6:10, 11:15] = True
    m[12:15, 13:19] = True
    p = ResidualBallProjectionEstimator().measure(evidence_from_mask(m))
    assert p.valid
    assert p.support_pixels == int(m.sum())
    assert p.center_xy_local is not None
    assert p.area_equivalent_diameter_px > 0
    assert p.moment_major_diameter_px >= p.moment_minor_diameter_px >= 0


def test_weighted_center_is_subpixel_and_follows_difference_energy():
    m = np.zeros((9, 9), dtype=bool)
    m[4, 3:6] = True
    s = np.zeros((9, 9), dtype=np.uint8)
    s[4, 3] = 40
    s[4, 4] = 100
    s[4, 5] = 240
    p = ResidualBallProjectionEstimator().measure(evidence_from_mask(m, s))
    assert p.valid
    cx, cy = p.center_xy_local
    assert 4.4 < cx < 5.0
    assert math.isclose(cy, 4.0, abs_tol=1e-9)


def test_roi_origin_produces_global_image_center():
    m = np.zeros((11, 11), dtype=bool)
    m[4:7, 4:7] = True
    p = ResidualBallProjectionEstimator().measure(
        evidence_from_mask(m), roi_origin_xy=(100.25, 200.5)
    )
    assert p.valid
    lx, ly = p.center_xy_local
    gx, gy = p.center_xy_global
    assert math.isclose(gx, 100.25 + lx, abs_tol=1e-9)
    assert math.isclose(gy, 200.5 + ly, abs_tol=1e-9)


def test_area_equivalent_diameter_matches_support_area_definition():
    m = np.zeros((15, 15), dtype=bool)
    m[5:10, 5:10] = True
    p = ResidualBallProjectionEstimator().measure(evidence_from_mask(m))
    expected = 2.0 * math.sqrt(25.0 / math.pi)
    assert math.isclose(p.area_equivalent_diameter_px, expected, rel_tol=1e-12)


def test_empty_or_rejected_residual_is_invalid_projection():
    m = np.zeros((9, 9), dtype=bool)
    ev = evidence_from_mask(m)
    p = ResidualBallProjectionEstimator().measure(ev)
    assert p.valid is False
    assert p.center_xy_local is None
    assert p.apparent_scale_px is None


def test_apparent_scale_is_image_evidence_not_xyz():
    m = np.zeros((25, 25), dtype=bool)
    m[10:15, 7:19] = True
    p = ResidualBallProjectionEstimator().measure(evidence_from_mask(m))
    assert p.valid
    assert p.apparent_scale_px is not None
    assert p.apparent_scale_px > 0
