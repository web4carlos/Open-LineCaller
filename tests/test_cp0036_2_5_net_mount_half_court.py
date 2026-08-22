from __future__ import annotations

from PIL import Image

from linecaller.dcf.camera_external_ownership import (
    CameraExternalOwnership,
)
from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
)
from linecaller.dcf.net_mount_half_court import (
    select_net_mount_half_court_area,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
    OfficialExternalLiveRuntime,
)
from linecaller.dcf.predicted_topk_selector import (
    PredictedTopKSelector,
)


def _full_calibration():
    return ExternalGridCalibrator(
        (
            (100.0, 500.0),
            (900.0, 500.0),
            (700.0, 100.0),
            (300.0, 100.0),
        ),
        (1000, 600),
        coverage="FULL_COURT",
        config=ExternalGridConfig(margin_bu=10.0),
    ).build(background_image="bg.png")


def _half_calibration():
    # Net-mounted phone convention:
    # near-left/right = net edge, far-right/left = baseline edge.
    return ExternalGridCalibrator(
        (
            (100.0, 500.0),
            (900.0, 500.0),
            (700.0, 100.0),
            (300.0, 100.0),
        ),
        (1000, 600),
        coverage="HALF_COURT",
        config=ExternalGridConfig(margin_bu=10.0),
    ).build(background_image="bg.png")


def _c(x: float, y: float, scale: float = 3.0):
    return BallComponent(
        bbox=(int(x) - 1, int(y) - 1, 3, 3),
        area=5,
        centroid_xy=(float(x), float(y)),
        scale_px=float(scale),
    )


def test_full_court_far_guard_rejects_opposite_half_bootstrap():
    cal = _full_calibration()
    model_selector = PredictedTopKSelector(
        calibration=cal,
        bootstrap_receiving_side_guard=True,
        receiving_side="FAR",
    )

    # top-view y~20 BU maps near upper part; y~160 maps lower/opposite half.
    top_to_img = model_selector._perspective_scale._top_to_image

    import cv2
    import numpy as np

    def project(x, y):
        p = np.asarray([[[x, y]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, top_to_img)[0, 0]
        return float(q[0]), float(q[1])

    far_xy = project(40.0, 20.0)
    near_xy = project(40.0, 160.0)

    assert model_selector._bootstrap_side_ok(_c(*far_xy))
    assert not model_selector._bootstrap_side_ok(_c(*near_xy))


def test_full_court_near_guard_is_symmetric():
    cal = _full_calibration()
    selector = PredictedTopKSelector(
        calibration=cal,
        bootstrap_receiving_side_guard=True,
        receiving_side="NEAR",
    )

    import cv2
    import numpy as np

    H = selector._perspective_scale._top_to_image

    def project(y):
        q = cv2.perspectiveTransform(
            np.asarray([[[40.0, y]]], dtype=np.float32),
            H,
        )[0, 0]
        return _c(float(q[0]), float(q[1]))

    assert not selector._bootstrap_side_ok(project(20.0))
    assert selector._bootstrap_side_ok(project(160.0))


def test_net_mount_half_court_area_has_three_boundaries_and_no_net_outside():
    cal = _half_calibration()
    area = select_net_mount_half_court_area(
        cal,
        camera_id="phone-a",
        mount_position="NET_LEFT",
        depth_bu=6.0,
    )

    assert area.active_cell_count > 0
    assert area.active_zones == (
        "HALF_LEFT",
        "HALF_BASELINE",
        "HALF_RIGHT",
    )

    court_y = cal.config.half_court_y_bu
    centers = []
    for cell in area.cells:
        x0, y0, x1, y1 = cell.top_view_rect_bu
        centers.append(((x0 + x1) / 2, (y0 + y1) / 2))

    # Nothing behind the net edge belongs to this phone.
    assert all(y <= court_y for _, y in centers)
    # Both sidelines and baseline are represented.
    assert any(x < 0 and 0 <= y <= court_y for x, y in centers)
    assert any(x > cal.config.court_x_bu and 0 <= y <= court_y for x, y in centers)
    assert any(y < 0 for _, y in centers)


def test_net_mount_mode_requires_half_court_calibration():
    cal = _full_calibration()
    try:
        select_net_mount_half_court_area(
            cal,
            camera_id="phone-a",
            mount_position="NET_LEFT",
            depth_bu=6.0,
        )
    except ValueError as exc:
        assert "HALF_COURT" in str(exc)
    else:
        raise AssertionError("FULL_COURT must fail closed in net-mount mode")


def test_official_config_defaults_historical_but_supports_net_mount_half():
    cfg = OfficialExternalLiveConfig()
    assert cfg.net_mount_half_court_mode is False
    assert cfg.trajectory_bootstrap_receiving_side_guard is False
    assert cfg.trajectory_bootstrap_net_margin_bu == 4.0


def test_half_runtime_uses_local_three_boundary_area():
    cal = _half_calibration()
    background = Image.new("RGB", cal.image_size, "black")
    ball = Image.new("RGB", (20, 20), (210, 255, 30))
    profile = LockedBallColorProfile.from_template(ball)
    ownership = CameraExternalOwnership(
        camera_id="phone-a",
        mount_position="NET_LEFT",
        zones=(
            "FAR_LEFT",
            "FAR_BASELINE",
            "FAR_RIGHT",
            "NEAR_LEFT",
            "NEAR_BASELINE",
            "NEAR_RIGHT",
        ),
        depth_bu=6.0,
    )
    runtime = OfficialExternalLiveRuntime(
        cal,
        background,
        profile,
        ownership,
        receiving_side="FAR",
        config=OfficialExternalLiveConfig(
            net_mount_half_court_mode=True,
        ),
    )
    assert runtime.active_external_cell_count > 0
    assert runtime._owned_area.active_zones == (
        "HALF_LEFT",
        "HALF_BASELINE",
        "HALF_RIGHT",
    )


def test_health_and_wizard_expose_cp0036_2_5():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    client = TestClient(app)
    health = client.get("/health").json()
    assert health["net_mount_half_court_feature_version"] == "CP-0036.2.5"

    html = client.get("/").text
    for token in (
        'id="bootSideRejects"',
        'id="bootFloorXY"',
        'id="bootScope"',
        'id="runtimeCourtMode"',
        "BOOT_SIDE_REJECT",
    ):
        assert token in html


def test_cp0036_2_5_structural_uniqueness():
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

    assert selector.count("def _bootstrap_side_ok(") == 1
    assert live.count(
        '"net_mount_half_court_feature_version": "CP-0036.2.5"'
    ) == 1
    assert wizard.count('id="bootSideRejects"') == 1
