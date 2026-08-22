from __future__ import annotations

import cv2
import numpy as np

from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibrator,
    ExternalGridConfig,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
)
from linecaller.dcf.predicted_topk_selector import (
    PredictedTopKSelector,
)


def _calibration(coverage: str = "HALF_COURT"):
    return ExternalGridCalibrator(
        (
            (100.0, 500.0),
            (900.0, 500.0),
            (700.0, 100.0),
            (300.0, 100.0),
        ),
        (1000, 600),
        coverage=coverage,
        config=ExternalGridConfig(margin_bu=10.0),
    ).build(background_image="bg.png")


def _component(x: float, y: float, scale: float = 3.0):
    return BallComponent(
        bbox=(int(x) - 1, int(y) - 1, 3, 3),
        area=5,
        centroid_xy=(float(x), float(y)),
        scale_px=float(scale),
    )


def _project(selector: PredictedTopKSelector, x_bu: float, y_bu: float):
    H = selector._perspective_scale._top_to_image
    q = cv2.perspectiveTransform(
        np.asarray([[[x_bu, y_bu]]], dtype=np.float32),
        H,
    )[0, 0]
    return _component(float(q[0]), float(q[1]))


def _player_selector():
    return PredictedTopKSelector(
        calibration=_calibration(),
        min_gate_px=1000.0,
        bootstrap_min_motion_px=0.5,
        bootstrap_min_straightness=0.50,
        bootstrap_receiving_side_guard=True,
        receiving_side="FAR",
        bootstrap_net_margin_bu=4.0,
        bootstrap_net_ingress_guard=True,
        bootstrap_ingress_depth_bu=36.0,
        bootstrap_min_inward_bu=1.0,
    )


def test_player_track_can_bootstrap_from_net_band_moving_inward():
    selector = _player_selector()
    c1 = _project(selector, 40.0, 88.0)
    c2 = _project(selector, 40.0, 82.0)
    c3 = _project(selector, 40.0, 76.0)

    result = selector.select(
        3,
        [c3],
        [(1, (c1,)), (2, (c2,))],
    )

    assert result.state == "BOOTSTRAP"
    assert result.history_depth >= 3
    assert result.bootstrap_ingress_rejections == 0
    assert result.bootstrap_ingress_reason == "ACCEPTED"
    assert result.bootstrap_ingress_delta_bu > 1.0


def test_player_track_cannot_bootstrap_deep_inside_half():
    selector = _player_selector()
    c1 = _project(selector, 40.0, 45.0)
    c2 = _project(selector, 40.0, 40.0)
    c3 = _project(selector, 40.0, 35.0)

    result = selector.select(
        3,
        [c3],
        [(1, (c1,)), (2, (c2,))],
    )

    assert result.state == "UNLOCKED"
    assert result.bootstrap_ingress_rejections >= 1
    assert result.bootstrap_ingress_reason == "START_NOT_NET_BAND"


def test_player_track_cannot_bootstrap_moving_back_toward_net():
    selector = _player_selector()
    c1 = _project(selector, 40.0, 84.0)
    c2 = _project(selector, 40.0, 87.0)
    c3 = _project(selector, 40.0, 90.0)

    result = selector.select(
        3,
        [c3],
        [(1, (c1,)), (2, (c2,))],
    )

    assert result.state == "UNLOCKED"
    assert result.bootstrap_ingress_rejections >= 1
    assert result.bootstrap_ingress_reason == "NOT_MOVING_INTO_HALF"
    assert result.bootstrap_ingress_delta_bu < 0.0


def test_ingress_guard_is_half_court_only():
    try:
        PredictedTopKSelector(
            calibration=_calibration("FULL_COURT"),
            bootstrap_net_ingress_guard=True,
        )
    except ValueError as exc:
        assert "HALF_COURT" in str(exc)
    else:
        raise AssertionError(
            "Facilities FULL_COURT must not use Player net-ingress guard"
        )


def test_historical_defaults_keep_ingress_guard_off():
    cfg = OfficialExternalLiveConfig()
    assert cfg.trajectory_net_ingress_guard is False
    assert cfg.trajectory_net_ingress_depth_bu == 36.0
    assert cfg.trajectory_net_ingress_min_inward_bu == 1.0


def test_runtime_rejects_ingress_without_player_half_mode():
    try:
        OfficialExternalLiveConfig(
            trajectory_net_ingress_guard=True,
            net_mount_half_court_mode=False,
        )
    except ValueError as exc:
        assert "net_mount_half_court_mode" in str(exc)
    else:
        raise AssertionError("Ingress guard must be Player half-court only")


def test_health_and_wizard_expose_cp0036_2_6():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    client = TestClient(app)
    health = client.get("/health").json()
    assert (
        health["net_ingress_acquisition_feature_version"]
        == "CP-0036.2.6"
    )

    html = client.get("/").text
    for token in (
        'id="bootIngressRejects"',
        'id="bootIngressStartY"',
        'id="bootIngressEndY"',
        'id="bootIngressDelta"',
        'id="bootIngressReason"',
        "BOOT_INGRESS_REJECT",
        "PLAYER NET HALF",
        "FACILITY FULL",
    ):
        assert token in html


def test_live_app_enables_ingress_only_for_half_court():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    live = (root / "linecaller/api/live_app.py").read_text(
        encoding="utf-8"
    )
    assert (
        "trajectory_net_ingress_guard=net_mount_half_court"
        in live
    )


def test_cp0036_2_6_structural_uniqueness():
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

    assert selector.count("def _bootstrap_ingress_ok(") == 1
    assert live.count(
        '"net_ingress_acquisition_feature_version": "CP-0036.2.6"'
    ) == 1
    assert wizard.count('id="bootIngressRejects"') == 1
