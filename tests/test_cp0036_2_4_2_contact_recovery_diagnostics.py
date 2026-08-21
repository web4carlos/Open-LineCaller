
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from linecaller.api.live_app import app
from linecaller.dcf.contact_appearance_recovery import (
    ContactRecoveryProjectedIdentityExternalGridFrameLoop,
)
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibrator,
    ExternalGridConfig,
    LockedBallColorProfile,
)
from linecaller.dcf.projected_z0_identity import ProjectedZ0SignatureBank


PTS = (
    (100.0, 500.0),
    (900.0, 500.0),
    (700.0, 100.0),
    (300.0, 100.0),
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
        config=ExternalGridConfig(margin_bu=6.0),
        image_up_unit=(0.0, -1.0),
    ).build(background_image="bg.png")


def _useful_signature(bank: ProjectedZ0SignatureBank):
    choices = []
    for cell_id in sorted(bank._by_cell):
        for signature in bank.signatures_for_cell(cell_id):
            if signature.anchor != "CENTER":
                continue
            x, y = signature.image_xy
            if (
                80 < x < SIZE[0] - 80
                and 100 < y < SIZE[1] - 100
                and signature.expected_diameter_px >= 4.0
            ):
                choices.append((y, signature))
    assert choices
    return max(choices, key=lambda item: item[0])[1]


def _paint(center, diameter, color=(245, 230, 40)) -> Image.Image:
    arr = np.asarray(_background(), dtype=np.uint8).copy()
    cx, cy = (int(round(center[0])), int(round(center[1])))
    r = max(2, int(round(float(diameter) / 2.0)))
    cv2.circle(arr, (cx, cy), r, color, -1, cv2.LINE_8)
    return Image.fromarray(arr, "RGB")


def _loop():
    cal = _calibration()
    profile = LockedBallColorProfile.from_template(_ball_template())
    loop = ContactRecoveryProjectedIdentityExternalGridFrameLoop(
        cal,
        _background(),
        profile,
        difference_threshold=10,
        min_color_pixels=2,
        min_floor_scale_ratio=0.35,
        max_floor_scale_ratio=2.0,
        min_outside_clearance_bu=0.20,
        min_up_bu=0.65,
        max_up_bu=2.25,
        max_after_frames=2,
        use_projected_signatures=True,
        require_approach_memory=True,
        approach_history_frames=3,
        approach_min_prior_observations=2,
        projection_anchor_distance_bu=0.95,
        approach_prediction_radius_bu=2.50,
        approach_min_total_motion_bu=0.65,
        contact_appearance_recovery=True,
        contact_recovery_max_frames=2,
        contact_recovery_prediction_radius_bu=1.60,
        contact_recovery_component_radius_bu=1.35,
        contact_recovery_min_down_bu_per_frame=0.05,
        contact_recovery_min_value_ratio=0.55,
    )
    return cal, profile, loop


def test_contact_recovery_restores_desaturated_contact_then_up():
    cal, profile, loop = _loop()
    signature = _useful_signature(ProjectedZ0SignatureBank(cal, profile))
    cx, cy = signature.image_xy
    d = max(4.0, float(signature.expected_diameter_px))

    p1 = (cx, cy - 2.0 * d)
    p2 = (cx, cy - 1.0 * d)
    p3 = (cx, cy)
    p4 = (cx, cy - 1.0 * d)

    loop.process_frame(1, _paint(p1, d))
    loop.process_frame(2, _paint(p2, d))
    contact = loop.process_frame(
        3,
        _paint(p3, d, color=(235, 235, 235)),
    )

    assert contact.ball_color_changed_pixels == 0
    assert contact.contact_recoveries >= 1
    assert contact.contact_recovery_frame == 3
    assert contact.contact_recovery_cell is not None
    assert contact.z0_candidates
    assert contact.approach_reason == "ACCEPTED"
    assert contact.candidate_cell == contact.z0_candidates[0].cell_id
    assert contact.projected_anchor in {
        "CENTER", "NORTH", "SOUTH", "EAST", "WEST"
    }
    assert "CONTACT_APPEARANCE_RECOVERY" in contact.stage_trace

    after = loop.process_frame(4, _paint(p4, d))
    assert after.up_confirmations


def test_desaturated_blob_without_incoming_locked_track_is_not_recovered():
    cal, profile, loop = _loop()
    signature = _useful_signature(ProjectedZ0SignatureBank(cal, profile))
    frame = _paint(
        signature.image_xy,
        signature.expected_diameter_px,
        color=(235, 235, 235),
    )
    result = loop.process_frame(1, frame)

    assert result.contact_recoveries == 0
    assert not result.z0_candidates


def test_contact_recovery_requires_downward_incoming_motion():
    cal, profile, loop = _loop()
    signature = _useful_signature(ProjectedZ0SignatureBank(cal, profile))
    cx, cy = signature.image_xy
    d = max(4.0, float(signature.expected_diameter_px))

    # Image-up is (0,-1). These locked observations move upward, not down.
    loop.process_frame(1, _paint((cx, cy + 2.0 * d), d))
    loop.process_frame(2, _paint((cx, cy + 1.0 * d), d))
    result = loop.process_frame(
        3,
        _paint((cx, cy), d, color=(235, 235, 235)),
    )

    assert result.contact_recoveries == 0
    assert not result.z0_candidates


def test_approach_rejection_exposes_exact_reason_and_thresholds():
    cal, profile, loop = _loop()
    signature = _useful_signature(ProjectedZ0SignatureBank(cal, profile))
    frame = _paint(signature.image_xy, signature.expected_diameter_px)

    loop.process_frame(1, frame)
    loop.process_frame(2, frame)
    result = loop.process_frame(3, frame)

    assert result.projected_signature_matches >= 1
    assert result.approach_rejections >= 1
    assert result.approach_reason == "INSUFFICIENT_MOTION"
    assert result.approach_total_motion_px < result.approach_min_motion_px
    assert result.approach_allowed_error_px is not None
    assert result.candidate_cell is not None
    assert result.projected_anchor is not None


def test_health_and_wizard_publish_contact_recovery_diagnostics():
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["ball_identity_feature_version"] == "CP-0036.2.4"
    assert health["contact_recovery_feature_version"] == "CP-0036.2.4.2"

    html = client.get("/").text
    for token in (
        'id="contactRecoveries"',
        'id="recoveryFrame"',
        'id="approachReason"',
        'id="approachMotion"',
        'id="approachMinMotion"',
        'id="approachPredError"',
        'id="approachAllowedError"',
        'id="candidateCell"',
        'id="projectedAnchor"',
    ):
        assert token in html
