from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibrator,
    LockedBallColorProfile,
)
from linecaller.dcf.player_net_half_synthetic import (
    PlayerNetHalfSyntheticReference,
    SyntheticHalfCourtConfig,
)
from linecaller.dcf.player_real_video_validation import (
    ExpectedOutCall,
    FEATURE_VERSION,
    build_player_validation_runtime,
    production_player_validation_config,
    validate_player_frame_stream,
)


def _scene(side: str = "RIGHT"):
    return PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side(side)
    )


def _assets(scene):
    background = Image.fromarray(
        scene.background_rgb(),
        mode="RGB",
    )
    template = Image.fromarray(
        scene.ball_template_rgb(),
        mode="RGB",
    )
    profile = LockedBallColorProfile.from_template(template)
    return background, profile


def _full_stream(scene):
    return [
        (truth.frame_no, scene.render_frame(truth))
        for truth in scene.truth_frames()
    ]


def test_real_video_harness_full_synthetic_stream_truth_passes(tmp_path):
    scene = _scene("RIGHT")
    background, profile = _assets(scene)

    report = validate_player_frame_stream(
        _full_stream(scene),
        fps=scene.config.fps,
        calibration=scene.calibration,
        background_rgb=background,
        ball_profile=profile,
        mount_side="RIGHT",
        output_dir=tmp_path / "validation",
        expected_calls=(ExpectedOutCall(24, "OUT_LEFT"),),
        contact_tolerance_frames=2,
    )

    assert report.feature_version == FEATURE_VERSION
    assert report.status == "TRUTH_PASS"
    assert report.technical_pass is True
    assert report.truth_evaluated is True
    assert report.truth_passed is True
    assert not report.truth_failures
    assert len(report.official_calls) == 1
    assert report.official_calls[0].contact_frame == 24
    assert report.official_calls[0].region == "OUT_LEFT"
    assert report.outside_only_runtime is True
    assert Path(report.report_json).exists()
    assert Path(report.timeline_jsonl).exists()
    assert Path(report.timeline_csv).exists()
    assert Path(report.keyframe_dir).exists()


def test_real_video_harness_without_truth_does_not_claim_accuracy(tmp_path):
    scene = _scene("LEFT")
    background, profile = _assets(scene)
    report = validate_player_frame_stream(
        _full_stream(scene),
        fps=scene.config.fps,
        calibration=scene.calibration,
        background_rgb=background,
        ball_profile=profile,
        mount_side="LEFT",
        output_dir=tmp_path / "diagnostic",
    )
    assert report.status == "DIAGNOSTIC_COMPLETED"
    assert report.truth_evaluated is False
    assert report.truth_passed is None
    assert report.technical_pass is True


def test_real_video_harness_expect_no_out_passes_on_in_rally(tmp_path):
    scene = _scene("RIGHT")
    background, profile = _assets(scene)
    stream = [
        (truth.frame_no, scene.render_frame(truth))
        for truth in scene.truth_frames()
        if 37 <= truth.frame_no <= 64
    ]
    report = validate_player_frame_stream(
        stream,
        fps=scene.config.fps,
        calibration=scene.calibration,
        background_rgb=background,
        ball_profile=profile,
        mount_side="RIGHT",
        output_dir=tmp_path / "in_only",
        expect_no_out=True,
    )
    assert report.status == "TRUTH_PASS"
    assert report.truth_passed is True
    assert not report.official_calls


def test_real_video_harness_truth_failure_reports_missing_call(tmp_path):
    scene = _scene("RIGHT")
    background, profile = _assets(scene)
    report = validate_player_frame_stream(
        _full_stream(scene),
        fps=scene.config.fps,
        calibration=scene.calibration,
        background_rgb=background,
        ball_profile=profile,
        mount_side="RIGHT",
        output_dir=tmp_path / "wrong_truth",
        expected_calls=(ExpectedOutCall(24, "OUT_RIGHT"),),
    )
    assert report.status == "TRUTH_FAIL"
    assert report.truth_passed is False
    assert report.truth_failures


def test_real_video_harness_rejects_wrong_frame_size():
    scene = _scene("RIGHT")
    background, profile = _assets(scene)
    bad = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(
        ValueError,
        match="automatic resize/rotate is forbidden",
    ):
        validate_player_frame_stream(
            [(0, bad)],
            fps=30.0,
            calibration=scene.calibration,
            background_rgb=background,
            ball_profile=profile,
            mount_side="RIGHT",
        )


def test_real_video_harness_rejects_full_court_calibration():
    scene = _scene("RIGHT")
    background, profile = _assets(scene)
    calibrator = ExternalGridCalibrator(
        scene.config.image_points,
        scene.config.image_size,
        coverage=CalibrationCoverage.FULL_COURT,
    )
    full = calibrator.build(background_image="background.png")
    with pytest.raises(ValueError, match="HALF_COURT"):
        build_player_validation_runtime(
            full,
            background,
            profile,
            "RIGHT",
        )


def test_real_video_profile_keeps_player_net_ingress_contract():
    config = production_player_validation_config()
    assert config.net_mount_half_court_mode is True
    assert config.trajectory_net_ingress_guard is True
    assert config.predicted_top3_candidates is True
    assert config.projected_z0_signatures is True
    assert config.contact_appearance_recovery is True


def test_health_exposes_cp0036_2_9():
    from fastapi.testclient import TestClient
    from linecaller.api.live_app import app

    health = TestClient(app).get("/health").json()
    assert (
        health["player_real_video_validation_feature_version"]
        == "CP-0036.2.9"
    )


def test_cp0036_2_9_structural_invariants():
    root = Path(__file__).resolve().parents[1]
    module = (
        root / "linecaller/dcf/player_real_video_validation.py"
    ).read_text(encoding="utf-8")
    tool = (
        root / "tools/validate_player_real_video.py"
    ).read_text(encoding="utf-8")
    live = (
        root / "linecaller/api/live_app.py"
    ).read_text(encoding="utf-8")

    assert module.count('FEATURE_VERSION = "CP-0036.2.9"') == 1
    assert module.count(
        "automatic resize/rotate is forbidden"
    ) >= 1
    assert module.count(
        "CalibrationCoverage.HALF_COURT"
    ) >= 2
    assert module.count(
        'str(cell.region).startswith("OUT_")'
    ) == 1
    assert tool.count("--expect-out") >= 1
    assert tool.count("--expect-no-out") >= 1
    assert live.count(
        '"player_real_video_validation_feature_version": '
        '"CP-0036.2.9"'
    ) == 1
