from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from linecaller.dcf.camera_external_ownership import CameraExternalOwnership
from linecaller.dcf.external_grid_frame_loop import LockedBallColorProfile
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
    OfficialExternalLiveRuntime,
)
from linecaller.dcf.player_net_half_synthetic import (
    PlayerNetHalfSyntheticReference,
    SyntheticHalfCourtConfig,
)


FEATURE_VERSION = "CP-0036.2.8"


@dataclass(frozen=True)
class SyntheticOfficialCallEvidence:
    confirm_frame: int
    contact_frame: int
    region: str
    cell_id: int


@dataclass(frozen=True)
class PlayerNetHalfE2ETruthResult:
    feature_version: str
    mount_side: str
    passed: bool
    failures: tuple[str, ...]
    frame_count: int
    active_external_cells: int
    pose_safe_frames: int
    scan_suppressed_frames: tuple[int, ...]
    frames_with_ball_components: int
    bootstrap_frames: tuple[int, ...]
    ingress_accepted_frames: tuple[int, ...]
    first_deep_locked_frames: tuple[int, ...]
    reacquire_deep_locked_frames: tuple[int, ...]
    z0_frames: tuple[int, ...]
    up_contact_frames: tuple[int, ...]
    official_calls: tuple[SyntheticOfficialCallEvidence, ...]
    in_scenario_out_frames: tuple[int, ...]
    post_in_out_frames: tuple[int, ...]
    raw_z0_total: int
    up_total: int
    outside_only_runtime: bool
    player_image_down_gate_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["official_calls"] = [
            asdict(call) for call in self.official_calls
        ]
        return data


def _production_player_config() -> OfficialExternalLiveConfig:
    """Mirror the strict Wizard/API Player settings for deterministic E2E."""
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


def _build_runtime(
    scene: PlayerNetHalfSyntheticReference,
    mount_side: str,
) -> OfficialExternalLiveRuntime:
    side = str(mount_side).strip().upper()
    if side not in {"RIGHT", "LEFT"}:
        raise ValueError("mount_side must be RIGHT or LEFT")

    background = Image.fromarray(
        scene.background_rgb(),
        mode="RGB",
    )
    ball_template = Image.fromarray(
        scene.ball_template_rgb(),
        mode="RGB",
    )
    ball_profile = LockedBallColorProfile.from_template(ball_template)

    ownership = CameraExternalOwnership(
        camera_id=f"synthetic-player-{side.lower()}",
        mount_position=f"NET_{side}",  # type: ignore[arg-type]
        zones=(
            "FAR_LEFT",
            "FAR_BASELINE",
            "FAR_RIGHT",
        ),
        depth_bu=6.0,
    )

    return OfficialExternalLiveRuntime(
        scene.calibration,
        background,
        ball_profile,
        ownership,
        receiving_side="FAR",
        config=_production_player_config(),
    )


def run_player_net_half_e2e_truth_gate(
    mount_side: str = "RIGHT",
) -> PlayerNetHalfE2ETruthResult:
    """Run rendered synthetic frames through the real official Player stack.

    The synthetic truth is an engineering regression oracle. It never replaces
    real phone-on-net footage as the physical validation gate.
    """
    side = str(mount_side).strip().upper()
    scene = PlayerNetHalfSyntheticReference(
        SyntheticHalfCourtConfig.for_mount_side(side)
    )
    runtime = _build_runtime(scene, side)

    truths = scene.truth_frames()
    failures: list[str] = []
    scan_suppressed: list[int] = []
    bootstrap_frames: list[int] = []
    ingress_accepted_frames: list[int] = []
    first_deep_locked: list[int] = []
    reacquire_deep_locked: list[int] = []
    z0_frames: list[int] = []
    up_contact_frames: list[int] = []
    official_calls: list[SyntheticOfficialCallEvidence] = []
    in_out_frames: list[int] = []
    post_in_out_frames: list[int] = []

    pose_safe_frames = 0
    frames_with_components = 0
    raw_z0_total = 0
    up_total = 0
    out_rally_closed = False

    frame_loop = getattr(runtime, "_frame_loop", None)
    player_image_down_gate_enabled = bool(
        getattr(frame_loop, "approach_direction_gate", True)
    )
    runtime_calibration = getattr(runtime, "_runtime_calibration", None)
    outside_only_runtime = bool(
        runtime_calibration is not None
        and runtime_calibration.cells
        and all(
            str(cell.region).startswith("OUT_")
            for cell in runtime_calibration.cells
        )
    )

    for truth in truths:
        frame = scene.render_frame(truth)
        result = runtime.process_frame(
            truth.frame_no,
            frame,
            now_s=float(truth.frame_no) / float(scene.config.fps),
        )

        if getattr(result.pose.state, "value", str(result.pose.state)) == "SAFE":
            pose_safe_frames += 1
        if result.scan_suppressed:
            scan_suppressed.append(int(truth.frame_no))

        loop = result.frame_loop_result
        if loop is not None:
            if int(loop.raw_ball_components) > 0:
                frames_with_components += 1

            state = str(loop.trajectory_lock_state or "UNLOCKED")
            if state == "BOOTSTRAP":
                bootstrap_frames.append(int(truth.frame_no))
            if (
                loop.trajectory_bootstrap_ingress_reason == "ACCEPTED"
            ):
                ingress_accepted_frames.append(int(truth.frame_no))

            if (
                truth.scenario == "DEEP_DISTRACTOR_ONLY"
                and state != "UNLOCKED"
            ):
                first_deep_locked.append(int(truth.frame_no))
            if (
                truth.scenario == "DEEP_DISTRACTOR_REACQUIRE"
                and state != "UNLOCKED"
            ):
                reacquire_deep_locked.append(int(truth.frame_no))

            if loop.z0_candidates:
                z0_frames.append(int(truth.frame_no))
            raw_z0_total += int(len(loop.z0_candidates))

            for confirmation in loop.up_confirmations:
                up_contact_frames.append(
                    int(confirmation.contact_frame)
                )
                up_total += 1

        for call in result.out_calls:
            evidence = SyntheticOfficialCallEvidence(
                confirm_frame=int(call.frame_no),
                contact_frame=int(call.contact_frame),
                region=str(call.region),
                cell_id=int(call.cell_id),
            )
            official_calls.append(evidence)
            if truth.scenario == "INGRESS_IN_COURT":
                in_out_frames.append(int(truth.frame_no))
            if int(truth.frame_no) >= 52:
                post_in_out_frames.append(int(truth.frame_no))

        # A confirmed OUT ends that rally. Reset event state exactly as the
        # match/session layer would before the next rally. Remaining synthetic
        # post-bounce frames must still fail to reacquire because they do not
        # enter through the net band.
        if result.out_calls and not out_rally_closed:
            runtime.reset_event_state()
            out_rally_closed = True

    if scan_suppressed:
        failures.append(
            f"pose/scan suppressed on frames {scan_suppressed}"
        )
    if runtime.active_external_cell_count <= 0:
        failures.append("Player runtime owns zero external cells")
    if not outside_only_runtime:
        failures.append("runtime calibration is not outside-only")
    if player_image_down_gate_enabled:
        failures.append(
            "Player still uses broadcast image-DOWN approach gate"
        )
    if frames_with_components <= 0:
        failures.append("rendered frames produced no ball components")

    if first_deep_locked:
        failures.append(
            "deep pre-ingress distractor acquired track on frames "
            f"{first_deep_locked}"
        )
    if reacquire_deep_locked:
        failures.append(
            "deep post-loss distractor reacquired track on frames "
            f"{reacquire_deep_locked}"
        )

    first_bootstrap = [
        frame
        for frame in bootstrap_frames
        if 9 <= frame <= 28
    ]
    second_bootstrap = [
        frame
        for frame in bootstrap_frames
        if 37 <= frame <= 57
    ]
    if not first_bootstrap:
        failures.append("OUT_LEFT rally never bootstrapped from net ingress")
    if not second_bootstrap:
        failures.append("IN rally never bootstrapped from net ingress")

    first_ingress_accept = [
        frame
        for frame in ingress_accepted_frames
        if 9 <= frame <= 28
    ]
    second_ingress_accept = [
        frame
        for frame in ingress_accepted_frames
        if 37 <= frame <= 57
    ]
    if not first_ingress_accept:
        failures.append("OUT_LEFT rally has no ACCEPTED ingress evidence")
    if not second_ingress_accept:
        failures.append("IN rally has no ACCEPTED ingress evidence")

    if 24 not in z0_frames:
        failures.append("truth contact frame 24 did not become official Z0")
    if 24 not in up_contact_frames:
        failures.append("truth contact frame 24 received no UP confirmation")

    if len(official_calls) != 1:
        failures.append(
            "expected exactly one official OUT call; "
            f"observed {len(official_calls)}"
        )
    else:
        call = official_calls[0]
        if call.contact_frame != 24:
            failures.append(
                "official OUT contact frame mismatch: "
                f"{call.contact_frame} != 24"
            )
        if call.region != "OUT_LEFT":
            failures.append(
                "official OUT region mismatch: "
                f"{call.region} != OUT_LEFT"
            )
        if not 25 <= call.confirm_frame <= 26:
            failures.append(
                "official OUT was not confirmed within the two-frame "
                f"UP window: {call.confirm_frame}"
            )

    if in_out_frames:
        failures.append(
            f"IN rally emitted OUT on frames {in_out_frames}"
        )
    if post_in_out_frames:
        failures.append(
            f"OUT emitted at/after IN truth contact: {post_in_out_frames}"
        )

    return PlayerNetHalfE2ETruthResult(
        feature_version=FEATURE_VERSION,
        mount_side=side,
        passed=not failures,
        failures=tuple(failures),
        frame_count=len(truths),
        active_external_cells=int(
            runtime.active_external_cell_count
        ),
        pose_safe_frames=int(pose_safe_frames),
        scan_suppressed_frames=tuple(scan_suppressed),
        frames_with_ball_components=int(frames_with_components),
        bootstrap_frames=tuple(bootstrap_frames),
        ingress_accepted_frames=tuple(ingress_accepted_frames),
        first_deep_locked_frames=tuple(first_deep_locked),
        reacquire_deep_locked_frames=tuple(reacquire_deep_locked),
        z0_frames=tuple(z0_frames),
        up_contact_frames=tuple(up_contact_frames),
        official_calls=tuple(official_calls),
        in_scenario_out_frames=tuple(in_out_frames),
        post_in_out_frames=tuple(post_in_out_frames),
        raw_z0_total=int(raw_z0_total),
        up_total=int(up_total),
        outside_only_runtime=bool(outside_only_runtime),
        player_image_down_gate_enabled=bool(
            player_image_down_gate_enabled
        ),
    )


def run_both_player_net_half_e2e_truth_gates() -> dict[str, Any]:
    results = {
        side: run_player_net_half_e2e_truth_gate(side)
        for side in ("RIGHT", "LEFT")
    }
    return {
        "feature_version": FEATURE_VERSION,
        "passed": all(result.passed for result in results.values()),
        "results": {
            side: result.to_dict()
            for side, result in results.items()
        },
        "note": (
            "Synthetic engineering regression only; real phone-on-net "
            "footage remains the physical validation gate."
        ),
    }
