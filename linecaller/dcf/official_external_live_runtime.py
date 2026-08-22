from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from linecaller.dcf.camera_external_ownership import (
    CameraExternalOwnership,
    OwnedExternalArea,
    ReceivingSide,
    calibration_for_camera_owned_area,
    select_camera_owned_external_area,
)
from linecaller.dcf.camera_pose_guard import (
    CameraPoseDecision,
    CameraPoseGuard,
    CameraPoseGuardConfig,
    PoseGuardState,
    apply_micro_adjustment,
)
from linecaller.dcf.external_grid_frame_loop import (
    ExternalFrameResult,
    ExternalGridCalibration,
    ExternalGridCell,
    ExternalGridFrameLoop,
    LockedBallColorProfile,
    UpConfirmation,
)
from linecaller.dcf.projected_z0_identity import (
    ProjectedIdentityExternalGridFrameLoop,
)
from linecaller.dcf.contact_appearance_recovery import (
    ContactRecoveryProjectedIdentityExternalGridFrameLoop,
)
from linecaller.dcf.z0_floor_glow_renderer import render_soft_floor_glow


@dataclass(frozen=True)
class OfficialExternalLiveConfig:
    glow_hold_s: float = 2.5
    glow_strength: float = 0.68

    difference_threshold: int = 18
    min_color_pixels: int = 2
    min_floor_scale_ratio: float = 0.45
    max_floor_scale_ratio: float = 1.80
    min_up_bu: float = 0.65
    max_up_bu: float = 2.25
    max_lateral_bu: float = 1.00
    min_after_scale_ratio: float = 0.45
    max_after_scale_ratio: float = 1.80
    max_after_frames: int = 2
    bingo_cooldown_frames: int = 6
    motion_merge_gap_px: float = 8.0
    min_outside_clearance_bu: float = 0.20

    # Historical runtime defaults remain unchanged. The Wizard/API opts into
    # these CP-0036.2.4 gates explicitly.
    projected_z0_signatures: bool = False
    require_approach_memory: bool = False
    approach_history_frames: int = 3
    approach_min_prior_observations: int = 2
    projection_anchor_distance_bu: float = 0.95
    approach_prediction_radius_bu: float = 2.50
    approach_min_total_motion_bu: float = 0.65

    # Historical defaults stay OFF. Wizard/API opts into CP-0036.2.4.2.
    contact_appearance_recovery: bool = False
    contact_recovery_max_frames: int = 2
    contact_recovery_prediction_radius_bu: float = 1.60
    contact_recovery_component_radius_bu: float = 1.35
    contact_recovery_min_down_bu_per_frame: float = 0.05
    contact_recovery_min_value_ratio: float = 0.55
    contact_recovery_max_candidates: int = 3

    # CP-0036.2.4.6. Historical default stays OFF; Wizard/API opts in.
    predicted_top3_candidates: bool = False
    trajectory_candidate_top_k: int = 3
    trajectory_candidate_min_gate_px: float = 5.0
    trajectory_candidate_gate_scale: float = 4.75
    trajectory_bootstrap_min_straightness: float = 0.70
    trajectory_bootstrap_scale_guard: bool = False
    trajectory_bootstrap_min_scale_ratio: float = 0.35
    trajectory_bootstrap_max_scale_ratio: float = 2.20
    trajectory_lock_max_misses: int = 2

    def __post_init__(self) -> None:
        if self.glow_hold_s <= 0.0:
            raise ValueError("glow_hold_s must be > 0")
        if not 0.0 <= self.glow_strength <= 1.0:
            raise ValueError("glow_strength must be in [0,1]")
        if self.motion_merge_gap_px < 0.0:
            raise ValueError("motion_merge_gap_px must be >= 0")
        if self.min_outside_clearance_bu < 0.0:
            raise ValueError("min_outside_clearance_bu must be >= 0")
        if self.approach_history_frames < 2:
            raise ValueError("approach_history_frames must be >= 2")
        if not (
            1
            <= self.approach_min_prior_observations
            <= self.approach_history_frames
        ):
            raise ValueError(
                "approach_min_prior_observations must be in "
                "[1, approach_history_frames]"
            )
        if self.projection_anchor_distance_bu <= 0.0:
            raise ValueError("projection_anchor_distance_bu must be > 0")
        if self.approach_prediction_radius_bu <= 0.0:
            raise ValueError("approach_prediction_radius_bu must be > 0")
        if self.approach_min_total_motion_bu <= 0.0:
            raise ValueError("approach_min_total_motion_bu must be > 0")
        if self.contact_recovery_max_frames < 1:
            raise ValueError("contact_recovery_max_frames must be >= 1")
        if self.contact_recovery_prediction_radius_bu <= 0.0:
            raise ValueError(
                "contact_recovery_prediction_radius_bu must be > 0"
            )
        if self.contact_recovery_component_radius_bu <= 0.0:
            raise ValueError(
                "contact_recovery_component_radius_bu must be > 0"
            )
        if self.contact_recovery_min_down_bu_per_frame < 0.0:
            raise ValueError(
                "contact_recovery_min_down_bu_per_frame must be >= 0"
            )
        if not 0.0 < self.contact_recovery_min_value_ratio <= 1.0:
            raise ValueError(
                "contact_recovery_min_value_ratio must be in (0,1]"
            )
        if self.contact_recovery_max_candidates < 1:
            raise ValueError(
                "contact_recovery_max_candidates must be >= 1"
            )
        if self.trajectory_candidate_top_k < 1:
            raise ValueError(
                "trajectory_candidate_top_k must be >= 1"
            )
        if self.trajectory_candidate_min_gate_px <= 0.0:
            raise ValueError(
                "trajectory_candidate_min_gate_px must be > 0"
            )
        if self.trajectory_candidate_gate_scale <= 0.0:
            raise ValueError(
                "trajectory_candidate_gate_scale must be > 0"
            )
        if not (
            0.0
            < self.trajectory_bootstrap_min_straightness
            <= 1.0
        ):
            raise ValueError(
                "trajectory_bootstrap_min_straightness must be in (0,1]"
            )
        if self.trajectory_bootstrap_min_scale_ratio <= 0.0:
            raise ValueError(
                "trajectory_bootstrap_min_scale_ratio must be > 0"
            )
        if (
            self.trajectory_bootstrap_max_scale_ratio
            < self.trajectory_bootstrap_min_scale_ratio
        ):
            raise ValueError(
                "trajectory_bootstrap_max_scale_ratio must be >= "
                "trajectory_bootstrap_min_scale_ratio"
            )
        if self.trajectory_lock_max_misses < 0:
            raise ValueError(
                "trajectory_lock_max_misses must be >= 0"
            )


@dataclass(frozen=True)
class OfficialOutCall:
    camera_id: str
    receiving_side: ReceivingSide
    frame_no: int
    contact_frame: int
    cell_id: int
    region: str
    polygon_image: tuple[tuple[float, float], ...]
    contact_xy: tuple[float, float]
    above_xy: tuple[float, float]
    call: str = "OUT"


@dataclass(frozen=True)
class ActiveExternalGlow:
    cell_id: int
    polygon_image: tuple[tuple[float, float], ...]
    started_at_s: float
    expires_at_s: float


@dataclass(frozen=True)
class OfficialLiveFrameResult:
    frame_no: int
    camera_id: str
    receiving_side: ReceivingSide
    pose: CameraPoseDecision
    pose_epoch: int
    active_external_cells: int
    frame_loop_result: ExternalFrameResult | None
    out_calls: tuple[OfficialOutCall, ...]
    rendered_bgr: np.ndarray
    scan_suppressed: bool
    reason: str

    @property
    def has_out(self) -> bool:
        return bool(self.out_calls)


def _as_rgb_image(frame: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    a = np.asarray(frame, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("frame must be RGB HxWx3")
    return Image.fromarray(a, mode="RGB")


def _as_rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    return np.asarray(_as_rgb_image(frame), dtype=np.uint8)


class OfficialExternalLiveRuntime:
    """
    Production runtime for the OFFICIAL outside-only architecture.

    Per frame:
        CAMERA FRAME
          -> CAMERA POSE GUARD
          -> CAMERA OWNERSHIP + RECEIVING SIDE
          -> OFFICIAL EXTERNAL GRID v2
          -> PILLOW/COLOR COMPONENTS
          -> Z0 CANDIDATE
          -> UP AFTER THE FACT
          -> CONFIRMED OUT
          -> SOFT FLOOR GLOW
    """

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        background_rgb: Image.Image | np.ndarray,
        ball_profile: LockedBallColorProfile,
        ownership: CameraExternalOwnership,
        *,
        receiving_side: ReceivingSide | str,
        config: OfficialExternalLiveConfig | None = None,
        pose_config: CameraPoseGuardConfig | None = None,
    ) -> None:
        self.config = config or OfficialExternalLiveConfig()
        self.pose_config = pose_config or CameraPoseGuardConfig()
        self.ownership = ownership
        self.ball_profile = ball_profile

        self._calibration = calibration
        self._background = _as_rgb_image(background_rgb)
        if self._background.size != tuple(calibration.image_size):
            raise ValueError("background size does not match calibration")

        self._receiving_side = self._normalize_side(receiving_side)
        self._pose_epoch = 0
        self._glows: list[ActiveExternalGlow] = []

        self._pose_guard = CameraPoseGuard(
            self._calibration,
            self._background,
            config=self.pose_config,
        )

        self._owned_area: OwnedExternalArea
        self._runtime_calibration: ExternalGridCalibration
        self._frame_loop: ExternalGridFrameLoop | None
        self._cells_by_id: dict[int, ExternalGridCell]
        self._rebuild_owned_runtime()

    @staticmethod
    def _normalize_side(side: ReceivingSide | str) -> ReceivingSide:
        s = str(side).strip().upper()
        if s not in {"FAR", "NEAR"}:
            raise ValueError("receiving_side must be FAR or NEAR")
        return s  # type: ignore[return-value]

    @property
    def receiving_side(self) -> ReceivingSide:
        return self._receiving_side

    @property
    def pose_epoch(self) -> int:
        return self._pose_epoch

    @property
    def active_external_cell_count(self) -> int:
        return self._owned_area.active_cell_count

    @property
    def active_glows(self) -> tuple[ActiveExternalGlow, ...]:
        return tuple(self._glows)

    @property
    def calibration(self) -> ExternalGridCalibration:
        return self._calibration

    def _new_frame_loop(
        self,
        calibration: ExternalGridCalibration,
    ) -> ExternalGridFrameLoop:
        c = self.config
        common = dict(
            difference_threshold=c.difference_threshold,
            min_color_pixels=c.min_color_pixels,
            min_floor_scale_ratio=c.min_floor_scale_ratio,
            max_floor_scale_ratio=c.max_floor_scale_ratio,
            min_up_bu=c.min_up_bu,
            max_up_bu=c.max_up_bu,
            max_lateral_bu=c.max_lateral_bu,
            min_after_scale_ratio=c.min_after_scale_ratio,
            max_after_scale_ratio=c.max_after_scale_ratio,
            max_after_frames=c.max_after_frames,
            bingo_cooldown_frames=c.bingo_cooldown_frames,
            motion_merge_gap_px=c.motion_merge_gap_px,
            min_outside_clearance_bu=c.min_outside_clearance_bu,
        )

        if c.contact_appearance_recovery:
            return ContactRecoveryProjectedIdentityExternalGridFrameLoop(
                calibration,
                self._background,
                self.ball_profile,
                use_projected_signatures=c.projected_z0_signatures,
                require_approach_memory=c.require_approach_memory,
                approach_history_frames=c.approach_history_frames,
                approach_min_prior_observations=(
                    c.approach_min_prior_observations
                ),
                projection_anchor_distance_bu=(
                    c.projection_anchor_distance_bu
                ),
                approach_prediction_radius_bu=(
                    c.approach_prediction_radius_bu
                ),
                approach_min_total_motion_bu=(
                    c.approach_min_total_motion_bu
                ),
                contact_appearance_recovery=True,
                contact_recovery_max_frames=(
                    c.contact_recovery_max_frames
                ),
                contact_recovery_prediction_radius_bu=(
                    c.contact_recovery_prediction_radius_bu
                ),
                contact_recovery_component_radius_bu=(
                    c.contact_recovery_component_radius_bu
                ),
                contact_recovery_min_down_bu_per_frame=(
                    c.contact_recovery_min_down_bu_per_frame
                ),
                contact_recovery_min_value_ratio=(
                    c.contact_recovery_min_value_ratio
                ),
                contact_recovery_max_candidates=(
                    c.contact_recovery_max_candidates
                ),
                predicted_top3_candidates=(
                    c.predicted_top3_candidates
                ),
                trajectory_candidate_top_k=(
                    c.trajectory_candidate_top_k
                ),
                trajectory_candidate_min_gate_px=(
                    c.trajectory_candidate_min_gate_px
                ),
                trajectory_candidate_gate_scale=(
                    c.trajectory_candidate_gate_scale
                ),
                trajectory_bootstrap_min_straightness=(
                    c.trajectory_bootstrap_min_straightness
                ),
                trajectory_bootstrap_scale_guard=(
                    c.trajectory_bootstrap_scale_guard
                ),
                trajectory_bootstrap_min_scale_ratio=(
                    c.trajectory_bootstrap_min_scale_ratio
                ),
                trajectory_bootstrap_max_scale_ratio=(
                    c.trajectory_bootstrap_max_scale_ratio
                ),
                trajectory_lock_max_misses=(
                    c.trajectory_lock_max_misses
                ),
                **common,
            )

        if c.projected_z0_signatures or c.require_approach_memory:
            return ProjectedIdentityExternalGridFrameLoop(
                calibration,
                self._background,
                self.ball_profile,
                use_projected_signatures=c.projected_z0_signatures,
                require_approach_memory=c.require_approach_memory,
                approach_history_frames=c.approach_history_frames,
                approach_min_prior_observations=(
                    c.approach_min_prior_observations
                ),
                projection_anchor_distance_bu=(
                    c.projection_anchor_distance_bu
                ),
                approach_prediction_radius_bu=(
                    c.approach_prediction_radius_bu
                ),
                approach_min_total_motion_bu=(
                    c.approach_min_total_motion_bu
                ),
                **common,
            )

        return ExternalGridFrameLoop(
            calibration,
            self._background,
            self.ball_profile,
            **common,
        )

    def _rebuild_owned_runtime(self) -> None:
        self._owned_area = select_camera_owned_external_area(
            self._calibration,
            self.ownership,
            self._receiving_side,
        )
        self._runtime_calibration = calibration_for_camera_owned_area(
            self._calibration,
            self._owned_area,
        )
        self._cells_by_id = {
            int(c.cell_id): c for c in self._runtime_calibration.cells
        }
        if not self._runtime_calibration.cells:
            self._frame_loop = None
        else:
            self._frame_loop = self._new_frame_loop(
                self._runtime_calibration,
            )

    def reset_event_state(self) -> None:
        self._glows.clear()
        self._rebuild_owned_runtime()

    def set_receiving_side(
        self,
        side: ReceivingSide | str,
    ) -> None:
        s = self._normalize_side(side)
        if s == self._receiving_side:
            return
        self._receiving_side = s
        self._glows.clear()
        self._rebuild_owned_runtime()

    def _apply_pose_repair(
        self,
        decision: CameraPoseDecision,
    ) -> None:
        adjusted = apply_micro_adjustment(
            self._calibration,
            self._background,
            decision,
        )
        self._calibration = adjusted.calibration
        self._background = adjusted.background_rgb
        self._pose_epoch += 1
        self._glows.clear()
        self._pose_guard = CameraPoseGuard(
            self._calibration,
            self._background,
            config=self.pose_config,
        )
        self._rebuild_owned_runtime()

    def _out_calls_from_confirmations(
        self,
        frame_no: int,
        confirmations: Iterable[UpConfirmation],
        *,
        now_s: float,
    ) -> tuple[OfficialOutCall, ...]:
        calls: list[OfficialOutCall] = []
        for confirmation in confirmations:
            cell = self._cells_by_id.get(int(confirmation.cell_id))
            if cell is None:
                continue
            if not str(cell.region).startswith("OUT_"):
                continue

            call = OfficialOutCall(
                camera_id=self.ownership.camera_id,
                receiving_side=self._receiving_side,
                frame_no=int(frame_no),
                contact_frame=int(confirmation.contact_frame),
                cell_id=int(cell.cell_id),
                region=str(cell.region),
                polygon_image=cell.polygon_image,
                contact_xy=confirmation.contact_xy,
                above_xy=confirmation.above_xy,
            )
            calls.append(call)
            self._glows.append(
                ActiveExternalGlow(
                    cell_id=int(cell.cell_id),
                    polygon_image=cell.polygon_image,
                    started_at_s=float(now_s),
                    expires_at_s=float(now_s) + self.config.glow_hold_s,
                )
            )
        return tuple(calls)

    def _render(
        self,
        frame_rgb: np.ndarray,
        *,
        now_s: float,
        allow_glow: bool,
    ) -> np.ndarray:
        bgr = frame_rgb[:, :, ::-1].copy()
        if not allow_glow:
            return bgr

        live: list[ActiveExternalGlow] = []
        for glow in self._glows:
            if float(now_s) >= glow.expires_at_s:
                continue
            duration = max(1e-6, glow.expires_at_s - glow.started_at_s)
            age = max(0.0, float(now_s) - glow.started_at_s)
            remaining = max(0.0, glow.expires_at_s - float(now_s))
            # A confirmed OUT must illuminate immediately on its confirmation
            # frame.  The previous envelope started at exactly zero because
            # age == 0 for a newly-created glow, which made the OUT event valid
            # but visually invisible for that frame.
            fade_in_window = max(1e-6, min(0.18, 0.25 * duration))
            fade_in = min(1.0, 0.35 + 0.65 * (age / fade_in_window))
            fade_out = min(
                1.0,
                remaining / max(1e-6, min(0.35, 0.30 * duration)),
            )
            envelope = min(fade_in, fade_out, 1.0)
            strength = self.config.glow_strength * envelope
            if strength > 0.0:
                bgr = render_soft_floor_glow(
                    bgr,
                    glow.polygon_image,
                    strength=strength,
                )
            live.append(glow)
        self._glows = live
        return bgr

    def process_frame(
        self,
        frame_no: int,
        frame_rgb: Image.Image | np.ndarray,
        *,
        now_s: float,
    ) -> OfficialLiveFrameResult:
        rgb = _as_rgb_array(frame_rgb)
        pose = self._pose_guard.evaluate(rgb)

        if pose.state is PoseGuardState.BLOCKED:
            self._glows.clear()
            return OfficialLiveFrameResult(
                frame_no=int(frame_no),
                camera_id=self.ownership.camera_id,
                receiving_side=self._receiving_side,
                pose=pose,
                pose_epoch=self._pose_epoch,
                active_external_cells=self.active_external_cell_count,
                frame_loop_result=None,
                out_calls=(),
                rendered_bgr=self._render(rgb, now_s=now_s, allow_glow=False),
                scan_suppressed=True,
                reason="POSE_BLOCKED_NO_CALL",
            )

        if pose.state is PoseGuardState.MICRO_ADJUST:
            self._apply_pose_repair(pose)
            return OfficialLiveFrameResult(
                frame_no=int(frame_no),
                camera_id=self.ownership.camera_id,
                receiving_side=self._receiving_side,
                pose=pose,
                pose_epoch=self._pose_epoch,
                active_external_cells=self.active_external_cell_count,
                frame_loop_result=None,
                out_calls=(),
                rendered_bgr=self._render(rgb, now_s=now_s, allow_glow=False),
                scan_suppressed=True,
                reason="POSE_REPAIRED_RECHECK_REQUIRED",
            )

        if self._frame_loop is None:
            return OfficialLiveFrameResult(
                frame_no=int(frame_no),
                camera_id=self.ownership.camera_id,
                receiving_side=self._receiving_side,
                pose=pose,
                pose_epoch=self._pose_epoch,
                active_external_cells=0,
                frame_loop_result=None,
                out_calls=(),
                rendered_bgr=self._render(rgb, now_s=now_s, allow_glow=True),
                scan_suppressed=True,
                reason="NO_CAMERA_OWNERSHIP_FOR_RECEIVING_SIDE",
            )

        loop_result = self._frame_loop.process_frame(frame_no, rgb)
        out_calls = self._out_calls_from_confirmations(
            frame_no,
            loop_result.bingo_cells,
            now_s=now_s,
        )

        return OfficialLiveFrameResult(
            frame_no=int(frame_no),
            camera_id=self.ownership.camera_id,
            receiving_side=self._receiving_side,
            pose=pose,
            pose_epoch=self._pose_epoch,
            active_external_cells=self.active_external_cell_count,
            frame_loop_result=loop_result,
            out_calls=out_calls,
            rendered_bgr=self._render(rgb, now_s=now_s, allow_glow=True),
            scan_suppressed=False,
            reason="OUT_CONFIRMED" if out_calls else "SAFE_EXTERNAL_SCAN",
        )