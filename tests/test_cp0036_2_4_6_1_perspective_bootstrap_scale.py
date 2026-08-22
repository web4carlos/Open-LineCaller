from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from linecaller.api.live_app import app
from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
)
from linecaller.dcf.predicted_topk_selector import (
    PerspectiveBallScaleModel,
    PredictedTopKSelector,
)


PTS = (
    (100.0, 500.0),
    (900.0, 500.0),
    (700.0, 100.0),
    (300.0, 100.0),
)
SIZE = (1000, 600)


def _calibration():
    return ExternalGridCalibrator(
        PTS,
        SIZE,
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=10.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image="bg.png")


def _c(
    x: float,
    y: float,
    *,
    scale: float,
) -> BallComponent:
    w = max(2, int(round(scale)))
    return BallComponent(
        bbox=(int(round(x)) - w // 2, int(round(y)) - w // 2, w, w),
        area=max(4, w * w // 2),
        centroid_xy=(float(x), float(y)),
        scale_px=float(scale),
    )


def test_continuous_perspective_scale_is_available_without_interior_grid():
    model = PerspectiveBallScaleModel(_calibration())
    expected = model.expected_diameter_px((246.0, 168.0))
    assert expected is not None
    assert 4.0 < expected < 7.0


def test_large_smooth_object_cannot_bootstrap_when_ball_scale_guard_is_on():
    cal = _calibration()
    selector = PredictedTopKSelector(
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
        calibration=cal,
        bootstrap_scale_guard=True,
        bootstrap_min_scale_ratio=0.35,
        bootstrap_max_scale_ratio=2.20,
    )

    motion_frames = (
        (
            1,
            (
                _c(230.0, 150.0, scale=3.0),
                _c(240.0, 160.0, scale=14.0),
            ),
        ),
        (
            2,
            (
                _c(232.0, 152.0, scale=3.0),
                _c(243.0, 164.0, scale=14.0),
            ),
        ),
    )
    real = _c(234.0, 154.0, scale=3.0)
    large = _c(246.0, 168.0, scale=14.0)

    selection = selector.select(
        3,
        [real, large],
        motion_frames,
    )

    assert selection.state == "BOOTSTRAP"
    assert real in selection.selected
    assert large not in selection.selected
    assert selection.bootstrap_scale_rejections > 0
    assert selection.bootstrap_scale_observed_px == 14.0
    assert selection.bootstrap_scale_expected_px is not None
    assert selection.bootstrap_scale_ratio is not None
    assert selection.bootstrap_scale_ratio > 2.20


def test_guard_uses_tiny_ball_raster_tolerance_for_small_projected_ball():
    # Directly verify the configured max still accepts a 3px raster ball
    # when perspective says the ideal diameter is around 1.5px.
    from linecaller.dcf.external_grid_frame_loop import (
        rasterized_scale_compatible,
    )

    assert rasterized_scale_compatible(
        3.0,
        1.50,
        0.35,
        2.20,
    )
    assert not rasterized_scale_compatible(
        14.0,
        1.50,
        0.35,
        2.20,
    )


def test_historical_default_off_but_wizard_health_reports_cp0036_2_4_6_1():
    cfg = OfficialExternalLiveConfig()
    assert cfg.trajectory_bootstrap_scale_guard is False
    assert cfg.trajectory_bootstrap_min_scale_ratio == 0.35
    assert cfg.trajectory_bootstrap_max_scale_ratio == 2.20

    client = TestClient(app)
    health = client.get("/health").json()
    assert (
        health["perspective_bootstrap_scale_feature_version"]
        == "CP-0036.2.4.6.1"
    )

    html = client.get("/").text
    for token in (
        'id="bootScaleRejects"',
        'id="bootScaleObserved"',
        'id="bootScaleExpected"',
        'id="bootScaleRatio"',
        "BOOT_SCALE_REJECT",
    ):
        assert token in html


def test_cp0036_2_4_6_1_source_blocks_are_unique():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    selector = (
        root / "linecaller/dcf/predicted_topk_selector.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")
    wizard = (
        root / "linecaller/api/static/wizard.html"
    ).read_text(encoding="utf-8")

    assert selector.count("class PerspectiveBallScaleModel:") == 1
    assert selector.count("def _bootstrap_scale_ok(") == 1
    assert (
        live.count(
            '"perspective_bootstrap_scale_feature_version": '
            '"CP-0036.2.4.6.1"'
        )
        == 1
    )
    assert wizard.count('id="bootScaleRejects"') == 1
    assert wizard.count("BOOT_SCALE_REJECT x${bootScaleRejects}") == 1
