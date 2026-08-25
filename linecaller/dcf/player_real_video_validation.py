from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.camera_external_ownership import CameraExternalOwnership
from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibration,
    LockedBallColorProfile,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
    OfficialExternalLiveRuntime,
)


FEATURE_VERSION = "CP-0036.2.9"


@dataclass(frozen=True)
class ExpectedOutCall:
    contact_frame: int
    region: str

    def __post_init__(self) -> None:
        region = str(self.region).strip().upper()
        if not region.startswith("OUT_"):
            raise ValueError("expected OUT region must start with OUT_")
        object.__setattr__(self, "region", region)


@dataclass(frozen=True)
class RealVideoOfficialCall:
    confirm_frame: int
    contact_frame: int
    region: str
    cell_id: int


@dataclass(frozen=True)
class RealVideoFrameDiagnostic:
    frame_no: int
    time_s: float
    pose_state: str
    pose_reason: str
    scan_suppressed: bool
    runtime_reason: str
    raw_ball_components: int
    ball_footprints: int
    trajectory_lock_state: str | None
    trajectory_lock_depth: int
    ingress_reason: str | None
    ingress_start_y_bu: float | None
    ingress_end_y_bu: float | None
    ingress_delta_bu: float | None
    z0_count: int
    z0_regions: tuple[str, ...]
    up_count: int
    up_contact_frames: tuple[int, ...]
    out_count: int
    out_regions: tuple[str, ...]
    out_contact_frames: tuple[int, ...]
    scale_reason: str | None
    approach_reason: str | None
    keyframe_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerRealVideoValidationReport:
    feature_version: str
    status: str
    technical_pass: bool
    truth_evaluated: bool
    truth_passed: bool | None
    truth_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    mount_side: str
    fps: float
    image_size: tuple[int, int]
    first_frame: int | None
    last_frame: int | None
    processed_frames: int
    active_external_cells: int
    pose_counts: dict[str, int]
    scan_suppressed_frames: tuple[int, ...]
    frames_with_ball_components: int
    bootstrap_frames: tuple[int, ...]
    ingress_accepted_frames: tuple[int, ...]
    z0_frames: tuple[int, ...]
    up_frames: tuple[int, ...]
    official_calls: tuple[RealVideoOfficialCall, ...]
    expected_calls: tuple[ExpectedOutCall, ...]
    expect_no_out: bool
    contact_tolerance_frames: int
    outside_only_runtime: bool
    timeline_jsonl: str | None
    timeline_csv: str | None
    keyframe_dir: str | None
    report_json: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["official_calls"] = [
            asdict(call) for call in self.official_calls
        ]
        data["expected_calls"] = [
            asdict(call) for call in self.expected_calls
        ]
        return data


def production_player_validation_config() -> OfficialExternalLiveConfig:
    """Strict Player runtime profile used by the live Wizard/API."""
    return OfficialExternalLiveConfig(
        projected_z0_signatures=True,
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
        contact_recovery_max_candidates=3,
        predicted_top3_candidates=True,
        trajectory_candidate_top_k=3,
        trajectory_candidate_min_gate_px=5.0,
        trajectory_candidate_gate_scale=4.75,
        trajectory_bootstrap_min_straightness=0.70,
        trajectory_bootstrap_scale_guard=True,
        trajectory_bootstrap_min_scale_ratio=0.35,
        trajectory_bootstrap_max_scale_ratio=2.20,
        trajectory_bootstrap_receiving_side_guard=True,
        trajectory_bootstrap_net_margin_bu=4.0,
        trajectory_lock_max_misses=2,
        net_mount_half_court_mode=True,
        trajectory_net_ingress_guard=True,
        trajectory_net_ingress_depth_bu=36.0,
        trajectory_net_ingress_min_inward_bu=1.0,
    )


def load_player_validation_assets(
    calibration_path: str | Path,
    ball_template_path: str | Path,
    background_path: str | Path | None = None,
) -> tuple[
    ExternalGridCalibration,
    Image.Image,
    LockedBallColorProfile,
]:
    cal_path = Path(calibration_path).expanduser().resolve()
    ball_path = Path(ball_template_path).expanduser().resolve()
    if not cal_path.exists():
        raise ValueError(f"calibration does not exist: {cal_path}")
    if not ball_path.exists():
        raise ValueError(f"ball template does not exist: {ball_path}")

    calibration = ExternalGridCalibration.load(cal_path)
    if (
        CalibrationCoverage.parse(calibration.coverage)
        is not CalibrationCoverage.HALF_COURT
    ):
        raise ValueError(
            "Player real-video validation requires HALF_COURT calibration"
        )

    if background_path is None:
        bg_path = Path(calibration.background_image).expanduser()
        if not bg_path.is_absolute():
            bg_path = cal_path.parent / bg_path
    else:
        bg_path = Path(background_path).expanduser()
    bg_path = bg_path.resolve()

    if not bg_path.exists():
        raise ValueError(f"background does not exist: {bg_path}")

    try:
        background = Image.open(bg_path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"unable to read background: {bg_path}"
        ) from exc

    try:
        ball_template = Image.open(ball_path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"unable to read ball template: {ball_path}"
        ) from exc

    if background.size != tuple(calibration.image_size):
        raise ValueError(
            "background size does not match calibration: "
            f"{background.size} != {calibration.image_size}"
        )

    profile = LockedBallColorProfile.from_template(ball_template)
    return calibration, background, profile


def build_player_validation_runtime(
    calibration: ExternalGridCalibration,
    background_rgb: Image.Image | np.ndarray,
    ball_profile: LockedBallColorProfile,
    mount_side: str,
) -> OfficialExternalLiveRuntime:
    side = str(mount_side).strip().upper()
    if side not in {"RIGHT", "LEFT"}:
        raise ValueError("mount_side must be RIGHT or LEFT")
    if (
        CalibrationCoverage.parse(calibration.coverage)
        is not CalibrationCoverage.HALF_COURT
    ):
        raise ValueError(
            "Player real-video validation requires HALF_COURT calibration"
        )

    ownership = CameraExternalOwnership(
        camera_id=f"player-validation-{side.lower()}",
        mount_position=f"NET_{side}",  # type: ignore[arg-type]
        zones=(
            "FAR_LEFT",
            "FAR_BASELINE",
            "FAR_RIGHT",
        ),
        depth_bu=6.0,
    )
    return OfficialExternalLiveRuntime(
        calibration,
        background_rgb,
        ball_profile,
        ownership,
        receiving_side="FAR",
        config=production_player_validation_config(),
    )


def _pose_state_name(state: Any) -> str:
    return str(getattr(state, "value", state))


def _truth_check(
    observed: tuple[RealVideoOfficialCall, ...],
    expected: tuple[ExpectedOutCall, ...],
    *,
    expect_no_out: bool,
    contact_tolerance_frames: int,
) -> tuple[bool | None, tuple[str, ...]]:
    if expected and expect_no_out:
        raise ValueError(
            "expected OUT calls and expect_no_out are mutually exclusive"
        )
    tolerance = int(contact_tolerance_frames)
    if tolerance < 0:
        raise ValueError("contact_tolerance_frames must be >= 0")

    if not expected and not expect_no_out:
        return None, ()

    failures: list[str] = []
    if expect_no_out:
        if observed:
            failures.append(
                "expected zero OUT calls but observed "
                f"{len(observed)}: "
                + ", ".join(
                    f"{c.contact_frame}:{c.region}"
                    for c in observed
                )
            )
        return (not failures), tuple(failures)

    unmatched = set(range(len(observed)))
    for exp in expected:
        candidates = [
            index
            for index in unmatched
            if observed[index].region == exp.region
            and abs(
                observed[index].contact_frame
                - int(exp.contact_frame)
            )
            <= tolerance
        ]
        if not candidates:
            failures.append(
                "missing expected OUT "
                f"{exp.contact_frame}:{exp.region} "
                f"(tolerance +/-{tolerance} frames)"
            )
            continue
        chosen = min(
            candidates,
            key=lambda index: abs(
                observed[index].contact_frame
                - int(exp.contact_frame)
            ),
        )
        unmatched.remove(chosen)

    if unmatched:
        extras = [observed[index] for index in sorted(unmatched)]
        failures.append(
            "unexpected extra OUT call(s): "
            + ", ".join(
                f"{call.contact_frame}:{call.region}"
                for call in extras
            )
        )

    return (not failures), tuple(failures)


def _event_tags(
    diagnostic: RealVideoFrameDiagnostic,
) -> tuple[str, ...]:
    tags: list[str] = []
    if diagnostic.pose_state != "SAFE":
        tags.append(f"POSE_{diagnostic.pose_state}")
    if diagnostic.trajectory_lock_state == "BOOTSTRAP":
        tags.append("BOOTSTRAP")
    if diagnostic.ingress_reason == "ACCEPTED":
        tags.append("INGRESS")
    if diagnostic.z0_count:
        tags.append("Z0")
    if diagnostic.up_count:
        tags.append("UP")
    if diagnostic.out_count:
        tags.append("OUT")
    return tuple(tags)


def _save_keyframe(
    rendered_bgr: np.ndarray,
    diagnostic: RealVideoFrameDiagnostic,
    tags: tuple[str, ...],
    directory: Path,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    canvas = np.asarray(rendered_bgr, dtype=np.uint8).copy()
    lines = [
        f"frame {diagnostic.frame_no}  t={diagnostic.time_s:.3f}s",
        (
            f"pose={diagnostic.pose_state} "
            f"track={diagnostic.trajectory_lock_state or '-'}"
        ),
        (
            f"ingress={diagnostic.ingress_reason or '-'} "
            f"Z0={diagnostic.z0_count} UP={diagnostic.up_count} "
            f"OUT={diagnostic.out_count}"
        ),
    ]
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22

    tag = "_".join(tags)[:80]
    filename = f"frame_{diagnostic.frame_no:06d}_{tag}.png"
    path = directory / filename
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write keyframe: {path}")
    return str(path.resolve())


def _timeline_csv_row(
    item: RealVideoFrameDiagnostic,
) -> dict[str, Any]:
    data = item.to_dict()
    for key in (
        "z0_regions",
        "up_contact_frames",
        "out_regions",
        "out_contact_frames",
    ):
        value = data[key]
        data[key] = ";".join(str(v) for v in value)
    return data


def _write_outputs(
    report: PlayerRealVideoValidationReport,
    timeline: list[RealVideoFrameDiagnostic],
    output_dir: Path,
) -> PlayerRealVideoValidationReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "timeline.jsonl"
    csv_path = output_dir / "timeline.csv"
    report_path = output_dir / "report.json"

    with jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for item in timeline:
            handle.write(
                json.dumps(item.to_dict(), sort_keys=True)
                + "\n"
            )

    fieldnames = list(
        _timeline_csv_row(timeline[0]).keys()
        if timeline
        else RealVideoFrameDiagnostic.__dataclass_fields__.keys()
    )
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in timeline:
            writer.writerow(_timeline_csv_row(item))

    updated = PlayerRealVideoValidationReport(
        feature_version=report.feature_version,
        status=report.status,
        technical_pass=report.technical_pass,
        truth_evaluated=report.truth_evaluated,
        truth_passed=report.truth_passed,
        truth_failures=report.truth_failures,
        warnings=report.warnings,
        mount_side=report.mount_side,
        fps=report.fps,
        image_size=report.image_size,
        first_frame=report.first_frame,
        last_frame=report.last_frame,
        processed_frames=report.processed_frames,
        active_external_cells=report.active_external_cells,
        pose_counts=report.pose_counts,
        scan_suppressed_frames=report.scan_suppressed_frames,
        frames_with_ball_components=report.frames_with_ball_components,
        bootstrap_frames=report.bootstrap_frames,
        ingress_accepted_frames=report.ingress_accepted_frames,
        z0_frames=report.z0_frames,
        up_frames=report.up_frames,
        official_calls=report.official_calls,
        expected_calls=report.expected_calls,
        expect_no_out=report.expect_no_out,
        contact_tolerance_frames=report.contact_tolerance_frames,
        outside_only_runtime=report.outside_only_runtime,
        timeline_jsonl=str(jsonl.resolve()),
        timeline_csv=str(csv_path.resolve()),
        keyframe_dir=str((output_dir / "keyframes").resolve()),
        report_json=str(report_path.resolve()),
    )
    report_path.write_text(
        json.dumps(updated.to_dict(), indent=2),
        encoding="utf-8",
    )
    return updated


def validate_player_frame_stream(
    frames: Iterable[tuple[int, np.ndarray | Image.Image]],
    *,
    fps: float,
    calibration: ExternalGridCalibration,
    background_rgb: Image.Image | np.ndarray,
    ball_profile: LockedBallColorProfile,
    mount_side: str,
    output_dir: str | Path | None = None,
    expected_calls: Iterable[ExpectedOutCall] = (),
    expect_no_out: bool = False,
    contact_tolerance_frames: int = 2,
    max_keyframes: int = 120,
    reset_after_out: bool = True,
) -> PlayerRealVideoValidationReport:
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("fps must be a finite value > 0")
    if int(max_keyframes) < 0:
        raise ValueError("max_keyframes must be >= 0")

    side = str(mount_side).strip().upper()
    expected = tuple(expected_calls)
    if expected and expect_no_out:
        raise ValueError(
            "expected OUT calls and expect_no_out are mutually exclusive"
        )

    runtime = build_player_validation_runtime(
        calibration,
        background_rgb,
        ball_profile,
        side,
    )
    runtime_calibration = getattr(
        runtime,
        "_runtime_calibration",
        None,
    )
    outside_only = bool(
        runtime_calibration is not None
        and runtime_calibration.cells
        and all(
            str(cell.region).startswith("OUT_")
            for cell in runtime_calibration.cells
        )
    )
    if not outside_only:
        raise RuntimeError(
            "Player validation runtime is not outside-grid-only"
        )

    destination = (
        None
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    key_dir = (
        None
        if destination is None
        else destination / "keyframes"
    )
    if key_dir is not None:
        key_dir.mkdir(parents=True, exist_ok=True)

    timeline: list[RealVideoFrameDiagnostic] = []
    calls: list[RealVideoOfficialCall] = []
    pose_counts: dict[str, int] = {}
    scan_suppressed: list[int] = []
    bootstrap_frames: list[int] = []
    ingress_accepted: list[int] = []
    z0_frames: list[int] = []
    up_frames: list[int] = []

    frames_with_components = 0
    first_frame: int | None = None
    last_frame: int | None = None
    keyframes_written = 0

    width, height = calibration.image_size

    for raw_frame_no, raw in frames:
        frame_no = int(raw_frame_no)
        if isinstance(raw, Image.Image):
            rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(raw, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"frame {frame_no} must be RGB HxWx3"
            )
        actual_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        if actual_size != (int(width), int(height)):
            raise ValueError(
                f"frame {frame_no} size {actual_size} does not match "
                f"calibration {(width, height)}; "
                "automatic resize/rotate is forbidden"
            )

        if first_frame is None:
            first_frame = frame_no
        last_frame = frame_no

        result = runtime.process_frame(
            frame_no,
            rgb,
            now_s=float(frame_no) / fps_value,
        )
        pose_state = _pose_state_name(result.pose.state)
        pose_counts[pose_state] = pose_counts.get(pose_state, 0) + 1
        if result.scan_suppressed:
            scan_suppressed.append(frame_no)

        loop = result.frame_loop_result
        raw_components = 0
        footprints = 0
        track_state: str | None = None
        lock_depth = 0
        ingress_reason: str | None = None
        ingress_start: float | None = None
        ingress_end: float | None = None
        ingress_delta: float | None = None
        z0_regions: tuple[str, ...] = ()
        up_contacts: tuple[int, ...] = ()
        scale_reason: str | None = None
        approach_reason: str | None = None

        if loop is not None:
            raw_components = int(loop.raw_ball_components)
            footprints = int(loop.ball_footprints)
            if raw_components > 0:
                frames_with_components += 1

            track_state = (
                None
                if loop.trajectory_lock_state is None
                else str(loop.trajectory_lock_state)
            )
            lock_depth = int(loop.trajectory_lock_depth)
            ingress_reason = (
                None
                if loop.trajectory_bootstrap_ingress_reason is None
                else str(
                    loop.trajectory_bootstrap_ingress_reason
                )
            )
            ingress_start = (
                None
                if loop.trajectory_bootstrap_ingress_start_y_bu is None
                else float(
                    loop.trajectory_bootstrap_ingress_start_y_bu
                )
            )
            ingress_end = (
                None
                if loop.trajectory_bootstrap_ingress_end_y_bu is None
                else float(
                    loop.trajectory_bootstrap_ingress_end_y_bu
                )
            )
            ingress_delta = (
                None
                if loop.trajectory_bootstrap_ingress_delta_bu is None
                else float(
                    loop.trajectory_bootstrap_ingress_delta_bu
                )
            )
            scale_reason = (
                None
                if loop.scale_diag_reason is None
                else str(loop.scale_diag_reason)
            )
            approach_reason = (
                None
                if loop.approach_reason is None
                else str(loop.approach_reason)
            )

            if track_state == "BOOTSTRAP":
                bootstrap_frames.append(frame_no)
            if ingress_reason == "ACCEPTED":
                ingress_accepted.append(frame_no)
            if loop.z0_candidates:
                z0_frames.append(frame_no)
                z0_regions = tuple(
                    str(candidate.region)
                    for candidate in loop.z0_candidates
                )
            if loop.up_confirmations:
                up_frames.append(frame_no)
                up_contacts = tuple(
                    int(item.contact_frame)
                    for item in loop.up_confirmations
                )

        frame_calls: list[RealVideoOfficialCall] = []
        for call in result.out_calls:
            evidence = RealVideoOfficialCall(
                confirm_frame=int(call.frame_no),
                contact_frame=int(call.contact_frame),
                region=str(call.region),
                cell_id=int(call.cell_id),
            )
            frame_calls.append(evidence)
            calls.append(evidence)

        diagnostic = RealVideoFrameDiagnostic(
            frame_no=frame_no,
            time_s=float(frame_no) / fps_value,
            pose_state=pose_state,
            pose_reason=str(result.pose.reason),
            scan_suppressed=bool(result.scan_suppressed),
            runtime_reason=str(result.reason),
            raw_ball_components=raw_components,
            ball_footprints=footprints,
            trajectory_lock_state=track_state,
            trajectory_lock_depth=lock_depth,
            ingress_reason=ingress_reason,
            ingress_start_y_bu=ingress_start,
            ingress_end_y_bu=ingress_end,
            ingress_delta_bu=ingress_delta,
            z0_count=len(z0_regions),
            z0_regions=z0_regions,
            up_count=len(up_contacts),
            up_contact_frames=up_contacts,
            out_count=len(frame_calls),
            out_regions=tuple(
                item.region for item in frame_calls
            ),
            out_contact_frames=tuple(
                item.contact_frame for item in frame_calls
            ),
            scale_reason=scale_reason,
            approach_reason=approach_reason,
        )

        tags = _event_tags(diagnostic)
        if (
            key_dir is not None
            and tags
            and keyframes_written < int(max_keyframes)
        ):
            saved = _save_keyframe(
                result.rendered_bgr,
                diagnostic,
                tags,
                key_dir,
            )
            diagnostic = RealVideoFrameDiagnostic(
                frame_no=diagnostic.frame_no,
                time_s=diagnostic.time_s,
                pose_state=diagnostic.pose_state,
                pose_reason=diagnostic.pose_reason,
                scan_suppressed=diagnostic.scan_suppressed,
                runtime_reason=diagnostic.runtime_reason,
                raw_ball_components=diagnostic.raw_ball_components,
                ball_footprints=diagnostic.ball_footprints,
                trajectory_lock_state=diagnostic.trajectory_lock_state,
                trajectory_lock_depth=diagnostic.trajectory_lock_depth,
                ingress_reason=diagnostic.ingress_reason,
                ingress_start_y_bu=diagnostic.ingress_start_y_bu,
                ingress_end_y_bu=diagnostic.ingress_end_y_bu,
                ingress_delta_bu=diagnostic.ingress_delta_bu,
                z0_count=diagnostic.z0_count,
                z0_regions=diagnostic.z0_regions,
                up_count=diagnostic.up_count,
                up_contact_frames=diagnostic.up_contact_frames,
                out_count=diagnostic.out_count,
                out_regions=diagnostic.out_regions,
                out_contact_frames=diagnostic.out_contact_frames,
                scale_reason=diagnostic.scale_reason,
                approach_reason=diagnostic.approach_reason,
                keyframe_file=saved,
            )
            keyframes_written += 1

        timeline.append(diagnostic)

        if frame_calls and reset_after_out:
            runtime.reset_event_state()

    truth_passed, truth_failures = _truth_check(
        tuple(calls),
        expected,
        expect_no_out=bool(expect_no_out),
        contact_tolerance_frames=int(contact_tolerance_frames),
    )
    truth_evaluated = truth_passed is not None

    warnings: list[str] = []
    if not timeline:
        warnings.append("NO_FRAMES_PROCESSED")
    if scan_suppressed:
        warnings.append(
            f"SCAN_SUPPRESSED_ON_{len(scan_suppressed)}_FRAME(S)"
        )
    if timeline and frames_with_components == 0:
        warnings.append("NO_SELECTED_BALL_COMPONENTS_DETECTED")
    if (
        frames_with_components > 0
        and not bootstrap_frames
    ):
        warnings.append("NO_NET_INGRESS_BOOTSTRAP")
    if (
        bootstrap_frames
        and not ingress_accepted
    ):
        warnings.append("NO_ACCEPTED_NET_INGRESS_TELEMETRY")

    technical_pass = bool(timeline) and outside_only
    if truth_evaluated:
        status = "TRUTH_PASS" if truth_passed else "TRUTH_FAIL"
    else:
        status = "DIAGNOSTIC_COMPLETED"

    report = PlayerRealVideoValidationReport(
        feature_version=FEATURE_VERSION,
        status=status,
        technical_pass=technical_pass,
        truth_evaluated=truth_evaluated,
        truth_passed=truth_passed,
        truth_failures=truth_failures,
        warnings=tuple(warnings),
        mount_side=side,
        fps=fps_value,
        image_size=(int(width), int(height)),
        first_frame=first_frame,
        last_frame=last_frame,
        processed_frames=len(timeline),
        active_external_cells=int(
            runtime.active_external_cell_count
        ),
        pose_counts=dict(pose_counts),
        scan_suppressed_frames=tuple(scan_suppressed),
        frames_with_ball_components=int(
            frames_with_components
        ),
        bootstrap_frames=tuple(bootstrap_frames),
        ingress_accepted_frames=tuple(ingress_accepted),
        z0_frames=tuple(z0_frames),
        up_frames=tuple(up_frames),
        official_calls=tuple(calls),
        expected_calls=expected,
        expect_no_out=bool(expect_no_out),
        contact_tolerance_frames=int(
            contact_tolerance_frames
        ),
        outside_only_runtime=outside_only,
        timeline_jsonl=None,
        timeline_csv=None,
        keyframe_dir=(
            None
            if key_dir is None
            else str(key_dir.resolve())
        ),
        report_json=None,
    )

    if destination is not None:
        report = _write_outputs(
            report,
            timeline,
            destination,
        )

    return report


def iter_video_rgb_frames(
    video_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[
    Iterator[tuple[int, np.ndarray]],
    float,
    tuple[int, int],
    int,
]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"unable to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if not np.isfinite(fps) or fps <= 0.0:
        cap.release()
        raise ValueError(
            "video FPS metadata is invalid; validation refuses "
            "to invent a frame rate"
        )
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("video dimensions are invalid")

    start = int(start_frame)
    if start < 0:
        cap.release()
        raise ValueError("start_frame must be >= 0")
    stop = None if end_frame is None else int(end_frame)
    if stop is not None and stop < start:
        cap.release()
        raise ValueError("end_frame must be >= start_frame")

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))

    def generator() -> Iterator[tuple[int, np.ndarray]]:
        frame_no = start
        try:
            while True:
                if stop is not None and frame_no > stop:
                    break
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                rgb = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB,
                )
                yield frame_no, rgb
                frame_no += 1
        finally:
            cap.release()

    return generator(), fps, (width, height), total


def validate_player_real_video(
    *,
    video_path: str | Path,
    calibration_path: str | Path,
    ball_template_path: str | Path,
    background_path: str | Path | None,
    mount_side: str,
    output_dir: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    expected_calls: Iterable[ExpectedOutCall] = (),
    expect_no_out: bool = False,
    contact_tolerance_frames: int = 2,
    max_keyframes: int = 120,
) -> PlayerRealVideoValidationReport:
    calibration, background, profile = (
        load_player_validation_assets(
            calibration_path,
            ball_template_path,
            background_path,
        )
    )
    frames, fps, video_size, _ = iter_video_rgb_frames(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if tuple(video_size) != tuple(calibration.image_size):
        if hasattr(frames, "close"):
            frames.close()  # type: ignore[attr-defined]
        raise ValueError(
            "video size does not match calibration: "
            f"{video_size} != {calibration.image_size}; "
            "automatic resize/rotate is forbidden"
        )

    return validate_player_frame_stream(
        frames,
        fps=fps,
        calibration=calibration,
        background_rgb=background,
        ball_profile=profile,
        mount_side=mount_side,
        output_dir=output_dir,
        expected_calls=expected_calls,
        expect_no_out=expect_no_out,
        contact_tolerance_frames=contact_tolerance_frames,
        max_keyframes=max_keyframes,
    )
