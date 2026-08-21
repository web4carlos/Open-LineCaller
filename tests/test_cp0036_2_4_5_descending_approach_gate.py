from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from linecaller.api.live_app import app
from linecaller.dcf.contact_appearance_recovery import (
    ContactRecoveryProjectedIdentityExternalGridFrameLoop,
)
from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
)


PTS = (
    (100.0, 500.0),
    (900.0, 500.0),
    (700.0, 100.0),
    (300.0, 100.0),
)
SIZE = (1000, 600)


def _ball_template() -> Image.Image:
    a = np.zeros((21, 21, 3), dtype=np.uint8)
    cv2.circle(a, (10, 10), 6, (245, 230, 40), -1, cv2.LINE_8)
    return Image.fromarray(a, "RGB")


def _loop() -> ContactRecoveryProjectedIdentityExternalGridFrameLoop:
    cal = ExternalGridCalibrator(
        PTS,
        SIZE,
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=10.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image="bg.png")
    return ContactRecoveryProjectedIdentityExternalGridFrameLoop(
        cal,
        Image.new("RGB", SIZE, (20, 20, 20)),
        LockedBallColorProfile.from_template(_ball_template()),
        approach_direction_gate=True,
        approach_min_down_bu_per_frame=0.05,
    )


def _c(x: float, y: float) -> BallComponent:
    return BallComponent(
        bbox=(int(x) - 1, int(y) - 1, 3, 3),
        area=7,
        centroid_xy=(float(x), float(y)),
        scale_px=3.0,
    )


def test_downward_approach_is_positive_in_calibrated_direction():
    loop = _loop()
    loop._motion.remember(10, [_c(200.0, 100.0)])
    loop._motion.remember(11, [_c(200.2, 102.0)])
    loop._motion.remember(12, [_c(200.4, 104.0)])

    down, minimum = loop._approach_down_rate(
        13,
        _c(200.6, 106.0),
        expected_floor_diameter_px=2.0,
    )
    assert down is not None
    assert down > 1.5
    assert down > minimum


def test_upward_false_track_fails_descending_threshold():
    loop = _loop()
    loop._motion.remember(10, [_c(286.0, 60.0)])
    loop._motion.remember(11, [_c(286.5, 58.5)])
    loop._motion.remember(12, [_c(287.0, 57.0)])

    down, minimum = loop._approach_down_rate(
        13,
        _c(287.5, 55.5),
        expected_floor_diameter_px=1.5,
    )
    assert down is not None
    assert down < 0.0
    assert down < minimum


def test_minimum_direction_threshold_scales_with_projected_ball():
    loop = _loop()
    loop._motion.remember(10, [_c(100.0, 100.0)])
    loop._motion.remember(11, [_c(100.0, 101.0)])

    _, minimum = loop._approach_down_rate(
        12,
        _c(100.0, 102.0),
        expected_floor_diameter_px=1.5,
    )
    assert abs(minimum - 0.075) < 1e-9


def test_health_and_wizard_expose_direction_gate():
    client = TestClient(app)
    health = client.get("/health").json()
    assert (
        health["approach_direction_feature_version"]
        == "CP-0036.2.4.5"
    )

    html = client.get("/").text
    for token in (
        'id="directionRejects"',
        'id="approachDownRate"',
        'id="approachMinDown"',
        "DIRECTION_REJECT",
    ):
        assert token in html


def test_direction_gate_source_blocks_are_unique():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    recovery = (
        root / "linecaller/dcf/contact_appearance_recovery.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")
    wizard = (
        root / "linecaller/api/static/wizard.html"
    ).read_text(encoding="utf-8")

    assert recovery.count("def _approach_down_rate(") == 1
    assert recovery.count("approach_direction_gate: bool = True") == 1
    assert (
        live.count(
            '"approach_direction_feature_version": "CP-0036.2.4.5"'
        )
        == 1
    )
    assert wizard.count('id="directionRejects"') == 1
    assert wizard.count(
        "const directionRejects = Number("
    ) == 1
