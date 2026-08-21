from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app
from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
    Z0Candidate,
    rasterized_scale_bounds_px,
    rasterized_scale_compatible,
)
from linecaller.dcf.projected_z0_identity import ProjectedZ0SignatureBank


PTS = (
    (100.0, 500.0),
    (900.0, 500.0),
    # Strong perspective on the FAR edge deliberately produces projected
    # pickleball diameters below 3 px, matching the real 640x360 reference.
    (560.0, 100.0),
    (440.0, 100.0),
)
SIZE = (1000, 600)


def _background() -> Image.Image:
    return Image.new("RGB", SIZE, (20, 20, 20))


def _ball_template() -> Image.Image:
    a = np.zeros((21, 21, 3), dtype=np.uint8)
    cv2.circle(a, (10, 10), 6, (245, 230, 40), -1, cv2.LINE_8)
    return Image.fromarray(a, "RGB")


def _calibration():
    return ExternalGridCalibrator(
        PTS,
        SIZE,
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=10.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image="bg.png")


def test_reference_1_654px_ball_accepts_three_pixel_raster_but_not_four():
    expected = 1.6543329129527347
    lo, hi = rasterized_scale_bounds_px(expected, 0.45, 1.80)
    assert lo < 2.0
    assert hi > 3.0
    assert rasterized_scale_compatible(
        3.0, expected, 0.45, 1.80
    )
    assert not rasterized_scale_compatible(
        4.0, expected, 0.45, 1.80
    )


def test_normal_size_ball_keeps_legacy_proportional_scale_gate():
    expected = 4.0
    lo, hi = rasterized_scale_bounds_px(expected, 0.45, 1.80)
    assert lo == 1.8
    assert hi == 7.2
    assert rasterized_scale_compatible(
        7.2, expected, 0.45, 1.80
    )
    assert not rasterized_scale_compatible(
        7.21, expected, 0.45, 1.80
    )


def test_external_z0_gate_reports_quantized_accept_for_tiny_ball():
    cal = _calibration()
    profile = LockedBallColorProfile.from_template(_ball_template())

    from linecaller.dcf.external_grid_frame_loop import ExternalGridFrameLoop

    loop = ExternalGridFrameLoop(
        cal,
        _background(),
        profile,
        min_floor_scale_ratio=0.45,
        max_floor_scale_ratio=1.80,
        min_outside_clearance_bu=0.0,
    )

    tiny_candidates = [
        cell
        for cell in cal.cells
        if 1.40
        <= float(cell.expected_floor_ball_diameter_px)
        <= 1.66
    ]
    assert tiny_candidates
    tiny = min(
        tiny_candidates,
        key=lambda c: abs(
            float(c.expected_floor_ball_diameter_px) - 1.55
        ),
    )
    expected = float(tiny.expected_floor_ball_diameter_px)
    assert 1.40 <= expected <= 1.66

    poly = np.asarray(tiny.polygon_image, dtype=np.float32)
    center = tuple(float(v) for v in poly.mean(axis=0))
    observed = 3.0
    ratio = observed / expected
    assert ratio > 1.80
    assert rasterized_scale_compatible(
        observed, expected, 0.45, 1.80
    )

    comp = BallComponent(
        bbox=(int(center[0]) - 1, int(center[1]) - 1, 3, 3),
        area=7,
        centroid_xy=center,
        scale_px=observed,
    )
    candidates, _ = loop._find_z0_candidates(77, [comp])
    assert candidates
    assert loop._last_scale_quantized_accepts >= 1
    assert loop._last_scale_diag_reason == "QUANTIZED_ACCEPT"
    assert loop._last_scale_diag_cell == tiny.cell_id
    assert loop._last_raw_z0_candidates >= 1


def test_projected_signature_bank_uses_same_raster_tolerance():
    cal = _calibration()
    profile = LockedBallColorProfile.from_template(_ball_template())
    bank = ProjectedZ0SignatureBank(cal, profile)

    signatures = [
        sig
        for cell_id in bank._by_cell
        for sig in bank.signatures_for_cell(cell_id)
        if 1.40 <= sig.expected_diameter_px <= 1.66
        and 20 < sig.image_xy[0] < SIZE[0] - 20
        and 20 < sig.image_xy[1] < SIZE[1] - 20
    ]
    assert signatures
    sig = min(
        signatures,
        key=lambda x: abs(x.expected_diameter_px - 1.55),
    )
    expected = float(sig.expected_diameter_px)
    assert 3.0 / expected > bank.max_scale_ratio

    component = BallComponent(
        bbox=(0, 0, 3, 3),
        area=7,
        centroid_xy=sig.image_xy,
        scale_px=3.0,
    )
    candidate = Z0Candidate(
        frame_no=1,
        cell_id=sig.cell_id,
        region="OUT_LEFT",
        color_pixels=7,
        centroid_xy=sig.image_xy,
        observed_scale_px=3.0,
        expected_floor_scale_px=expected,
        scale_ratio=3.0 / expected,
        floor_xy_bu=sig.floor_xy_bu,
        boundary_clearance_bu=2.0,
    )
    assert bank.best_match(candidate, component) is not None

    too_large = BallComponent(
        bbox=(0, 0, 4, 4),
        area=12,
        centroid_xy=sig.image_xy,
        scale_px=4.0,
    )
    candidate4 = Z0Candidate(
        frame_no=1,
        cell_id=sig.cell_id,
        region="OUT_LEFT",
        color_pixels=12,
        centroid_xy=sig.image_xy,
        observed_scale_px=4.0,
        expected_floor_scale_px=expected,
        scale_ratio=4.0 / expected,
        floor_xy_bu=sig.floor_xy_bu,
        boundary_clearance_bu=2.0,
    )
    assert bank.best_match(candidate4, too_large) is None


def test_health_and_wizard_expose_cp0036_2_4_4_diagnostics():
    client = TestClient(app)
    health = client.get("/health").json()
    assert (
        health["tiny_ball_scale_feature_version"]
        == "CP-0036.2.4.4"
    )

    html = client.get("/").text
    for token in (
        'id="extCellHits"',
        'id="rawZ0Total"',
        'id="scaleLow"',
        'id="scaleHigh"',
        'id="rasterAccepts"',
        'id="scaleReason"',
        'id="scaleObserved"',
        'id="scaleExpected"',
        'id="scaleRatio"',
        "RASTER_SCALE_ACCEPT",
    ):
        assert token in html


def test_cp0036_2_4_4_source_blocks_are_not_duplicated():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    external = (
        root / "linecaller/dcf/external_grid_frame_loop.py"
    ).read_text(encoding="utf-8")
    live_app = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")
    wizard = (
        root / "linecaller/api/static/wizard.html"
    ).read_text(encoding="utf-8")

    assert external.count("def rasterized_scale_bounds_px(") == 1
    assert external.count("def rasterized_scale_compatible(") == 1
    assert external.count("def _reset_pre_z0_telemetry(") == 1
    assert external.count("def _record_scale_diagnostic(") == 1

    assert (
        live_app.count(
            '"tiny_ball_scale_feature_version": "CP-0036.2.4.4"'
        )
        == 1
    )

    assert wizard.count('id="extCellHits"') == 1
    assert wizard.count(
        "const extHits = Number(d.external_cell_component_hits || 0);"
    ) == 1
    assert wizard.count("RASTER_SCALE_ACCEPT x${rasterAccepts}") == 1
