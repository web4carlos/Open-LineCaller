from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from linecaller.dcf.camera_external_ownership import CameraExternalOwnership
from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    LockedBallColorProfile,
)
from linecaller.dcf.official_external_live_runtime import (
    OfficialExternalLiveConfig,
    OfficialExternalLiveRuntime,
)


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNTIME_UPLOAD_DIR = Path(
    os.environ.get(
        "LINECALLER_RUNTIME_DIR",
        str(ROOT / ".linecaller_runtime" / "live"),
    )
).expanduser().resolve()


class PathConfigureRequest(BaseModel):
    calibration_path: str
    ball_template_path: str
    background_path: str | None = None
    camera_id: str = "camera-01"
    mount_position: str = "NET_CENTER"
    zones: list[str]
    receiving_side: str = "FAR"
    depth_bu: float = 6.0


class ReceivingSideRequest(BaseModel):
    side: str


@dataclass
class LiveSessionSummary:
    configured: bool = False
    camera_id: str | None = None
    mount_position: str | None = None
    zones: tuple[str, ...] = ()
    receiving_side: str | None = None
    calibration_path: str | None = None
    background_path: str | None = None
    ball_template_path: str | None = None
    image_size: tuple[int, int] | None = None
    active_external_cells: int = 0
    pose_epoch: int = 0


class LiveRuntimeRegistry:
    def __init__(self) -> None:
        self.runtime: OfficialExternalLiveRuntime | None = None
        self.summary = LiveSessionSummary()
        self._frame_no = 0

    @staticmethod
    def _resolve_background(
        calibration_path: Path,
        calibration: ExternalGridCalibration,
        explicit_background: str | Path | None,
    ) -> Path:
        if explicit_background is not None:
            p = Path(explicit_background).expanduser()
        else:
            p = Path(calibration.background_image).expanduser()

        if not p.is_absolute():
            p = calibration_path.parent / p
        return p.resolve()

    @staticmethod
    def _load_rgb(path: Path, label: str) -> Image.Image:
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to read {label}: {path}") from exc

    def configure_paths(
        self,
        *,
        calibration_path: str | Path,
        ball_template_path: str | Path,
        background_path: str | Path | None,
        camera_id: str,
        mount_position: str,
        zones: list[str] | tuple[str, ...],
        receiving_side: str,
        depth_bu: float,
    ) -> LiveSessionSummary:
        cal_path = Path(calibration_path).expanduser().resolve()
        ball_path = Path(ball_template_path).expanduser().resolve()

        if not cal_path.exists():
            raise ValueError(f"Calibration does not exist: {cal_path}")
        if not ball_path.exists():
            raise ValueError(f"Ball template does not exist: {ball_path}")

        calibration = ExternalGridCalibration.load(cal_path)
        bg_path = self._resolve_background(
            cal_path,
            calibration,
            background_path,
        )
        background = self._load_rgb(bg_path, "background image")
        ball_template = self._load_rgb(ball_path, "ball template")

        if background.size != tuple(calibration.image_size):
            raise ValueError(
                "Background image size "
                f"{background.size} does not match calibration "
                f"{calibration.image_size}"
            )

        ball_profile = LockedBallColorProfile.from_template(ball_template)
        ownership = CameraExternalOwnership(
            camera_id=camera_id,
            mount_position=mount_position,  # type: ignore[arg-type]
            zones=tuple(zones),  # type: ignore[arg-type]
            depth_bu=float(depth_bu),
        )

        runtime = OfficialExternalLiveRuntime(
            calibration,
            background,
            ball_profile,
            ownership,
            receiving_side=receiving_side,
            config=OfficialExternalLiveConfig(
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
                trajectory_lock_max_misses=2,
            ),
        )

        self.runtime = runtime
        self._frame_no = 0
        self.summary = LiveSessionSummary(
            configured=True,
            camera_id=ownership.camera_id,
            mount_position=ownership.mount_position,
            zones=tuple(ownership.zones),
            receiving_side=runtime.receiving_side,
            calibration_path=str(cal_path),
            background_path=str(bg_path),
            ball_template_path=str(ball_path),
            image_size=tuple(calibration.image_size),
            active_external_cells=runtime.active_external_cell_count,
            pose_epoch=runtime.pose_epoch,
        )
        return self.summary

    def set_receiving_side(self, side: str) -> LiveSessionSummary:
        if self.runtime is None:
            raise RuntimeError("Live runtime is not configured")
        self.runtime.set_receiving_side(side)
        self.summary.receiving_side = self.runtime.receiving_side
        self.summary.active_external_cells = (
            self.runtime.active_external_cell_count
        )
        self.summary.pose_epoch = self.runtime.pose_epoch
        return self.summary

    def next_frame_no(self) -> int:
        self._frame_no += 1
        return self._frame_no

    def sync_runtime_summary(self) -> None:
        if self.runtime is None:
            return
        self.summary.receiving_side = self.runtime.receiving_side
        self.summary.active_external_cells = (
            self.runtime.active_external_cell_count
        )
        self.summary.pose_epoch = self.runtime.pose_epoch


registry = LiveRuntimeRegistry()
app = FastAPI(
    title="Open LineCaller Live API",
    version="CP-0036.2.1",
    description="Official outside-grid live frame bridge.",
)


def _summary_dict() -> dict[str, Any]:
    registry.sync_runtime_summary()
    d = asdict(registry.summary)
    if d["image_size"] is not None:
        d["image_size"] = list(d["image_size"])
    d["zones"] = list(d["zones"])
    return d


def _save_upload(upload: UploadFile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path.resolve()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = STATIC_DIR / "wizard.html"
    if not path.exists():
        return HTMLResponse(
            "<h1>Open LineCaller</h1><p>Live wizard missing.</p>",
            status_code=500,
        )
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )


@app.get("/engineering", response_class=HTMLResponse)
def engineering_console() -> HTMLResponse:
    path = STATIC_DIR / "index.html"
    if not path.exists():
        return HTMLResponse(
            "<h1>Open LineCaller</h1><p>Engineering console missing.</p>",
            status_code=500,
        )
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "Open LineCaller Live API",
        "version": "CP-0036.2",
        "feature_version": "CP-0036.2.1",
        "runtime_feature_version": "CP-0036.2.3",
        "ball_identity_feature_version": "CP-0036.2.4",
        "contact_recovery_feature_version": "CP-0036.2.4.2",
        "wizard_session_feature_version": "CP-0036.2.4.3",
        "tiny_ball_scale_feature_version": "CP-0036.2.4.4",
        "approach_direction_feature_version": "CP-0036.2.4.5",
        "predicted_top3_feature_version": "CP-0036.2.4.6",
        "configured": registry.runtime is not None,
    }


@app.get("/api/live/status")
def live_status() -> dict[str, Any]:
    return _summary_dict()


@app.post("/api/live/configure-paths")
def configure_paths(payload: PathConfigureRequest) -> dict[str, Any]:
    try:
        registry.configure_paths(
            calibration_path=payload.calibration_path,
            ball_template_path=payload.ball_template_path,
            background_path=payload.background_path,
            camera_id=payload.camera_id,
            mount_position=payload.mount_position,
            zones=payload.zones,
            receiving_side=payload.receiving_side,
            depth_bu=payload.depth_bu,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _summary_dict()


@app.post("/api/live/configure-upload")
def configure_upload(
    calibration: UploadFile = File(...),
    background: UploadFile = File(...),
    ball_template: UploadFile = File(...),
    camera_id: str = Form("camera-01"),
    mount_position: str = Form("NET_CENTER"),
    zones: str = Form("FAR_LEFT,FAR_BASELINE,FAR_RIGHT"),
    receiving_side: str = Form("FAR"),
    depth_bu: float = Form(6.0),
) -> dict[str, Any]:
    session_dir = RUNTIME_UPLOAD_DIR
    session_dir.mkdir(parents=True, exist_ok=True)

    cal_suffix = Path(calibration.filename or "calibration.json").suffix or ".json"
    bg_suffix = Path(background.filename or "background.jpg").suffix or ".jpg"
    ball_suffix = Path(ball_template.filename or "ball.png").suffix or ".png"

    cal_path = _save_upload(
        calibration,
        session_dir / f"calibration{cal_suffix}",
    )
    bg_path = _save_upload(
        background,
        session_dir / f"background{bg_suffix}",
    )
    ball_path = _save_upload(
        ball_template,
        session_dir / f"ball_template{ball_suffix}",
    )

    zone_list = [z.strip().upper() for z in zones.split(",") if z.strip()]
    try:
        registry.configure_paths(
            calibration_path=cal_path,
            ball_template_path=ball_path,
            background_path=bg_path,
            camera_id=camera_id,
            mount_position=mount_position,
            zones=zone_list,
            receiving_side=receiving_side,
            depth_bu=depth_bu,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _summary_dict()


@app.post("/api/live/reset-events")
def reset_live_events() -> dict[str, Any]:
    runtime = registry.runtime
    if runtime is None:
        raise HTTPException(status_code=409, detail="Live runtime is not configured")
    runtime.reset_event_state()
    registry._frame_no = 0
    registry.sync_runtime_summary()
    return {
        "ok": True,
        "active_external_cells": runtime.active_external_cell_count,
        "receiving_side": runtime.receiving_side,
    }


@app.post("/api/live/receiving-side")
def set_receiving_side(payload: ReceivingSideRequest) -> dict[str, Any]:
    try:
        registry.set_receiving_side(payload.side)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _summary_dict()


@app.post("/api/live/frame")
async def process_frame(
    request: Request,
    frame_no: int | None = Query(default=None, ge=0),
    now_s: float | None = Query(default=None),
    include_image: bool = Query(default=True),
) -> dict[str, Any]:
    runtime = registry.runtime
    if runtime is None:
        raise HTTPException(
            status_code=409,
            detail="Live runtime is not configured",
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame body")

    encoded = np.frombuffer(body, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Frame is not a valid JPEG/PNG image",
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fn = registry.next_frame_no() if frame_no is None else int(frame_no)
    timestamp = time.monotonic() if now_s is None else float(now_s)

    try:
        result = runtime.process_frame(
            fn,
            rgb,
            now_s=timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry.sync_runtime_summary()

    out_calls = [
        {
            "call": call.call,
            "camera_id": call.camera_id,
            "receiving_side": call.receiving_side,
            "frame_no": call.frame_no,
            "contact_frame": call.contact_frame,
            "cell_id": call.cell_id,
            "region": call.region,
            "contact_xy": list(call.contact_xy),
            "above_xy": list(call.above_xy),
            "polygon_image": [list(q) for q in call.polygon_image],
        }
        for call in result.out_calls
    ]

    payload: dict[str, Any] = {
        "frame_no": result.frame_no,
        "camera_id": result.camera_id,
        "receiving_side": result.receiving_side,
        "pose_state": result.pose.state.value,
        "pose_reason": result.pose.reason,
        "pose_epoch": result.pose_epoch,
        "active_external_cells": result.active_external_cells,
        "scan_suppressed": result.scan_suppressed,
        "reason": result.reason,
        "has_out": result.has_out,
        "out_calls": out_calls,
    }

    if result.frame_loop_result is not None:
        loop = result.frame_loop_result
        payload["changed_pixels"] = loop.changed_pixels
        payload["ball_color_changed_pixels"] = loop.ball_color_changed_pixels
        payload["raw_ball_components"] = loop.raw_ball_components
        payload["ball_footprints"] = loop.ball_footprints
        payload["merged_motion_footprints"] = loop.merged_motion_footprints
        payload["boundary_guard_rejections"] = loop.boundary_guard_rejections
        payload["projected_signature_matches"] = loop.projected_signature_matches
        payload["projected_signature_rejections"] = loop.projected_signature_rejections
        payload["approach_rejections"] = loop.approach_rejections
        payload["motion_history_observations"] = loop.motion_history_observations
        payload["predicted_z0_cell"] = loop.predicted_z0_cell
        payload["approach_reason"] = loop.approach_reason
        payload["approach_direction_rejections"] = (
            loop.approach_direction_rejections
        )
        payload["approach_down_px_per_frame"] = (
            loop.approach_down_px_per_frame
        )
        payload["approach_min_down_px_per_frame"] = (
            loop.approach_min_down_px_per_frame
        )
        payload["trajectory_lock_state"] = loop.trajectory_lock_state
        payload["trajectory_components_considered"] = (
            loop.trajectory_components_considered
        )
        payload["trajectory_topk_selected"] = (
            loop.trajectory_topk_selected
        )
        payload["trajectory_candidate_rejections"] = (
            loop.trajectory_candidate_rejections
        )
        payload["trajectory_predicted_xy"] = (
            None
            if loop.trajectory_predicted_xy is None
            else list(loop.trajectory_predicted_xy)
        )
        payload["trajectory_gate_px"] = loop.trajectory_gate_px
        payload["trajectory_lock_depth"] = loop.trajectory_lock_depth
        payload["trajectory_topk_selected_xy"] = [
            list(point)
            for point in loop.trajectory_topk_selected_xy
        ]
        payload["trajectory_bootstrap_straightness"] = (
            loop.trajectory_bootstrap_straightness
        )
        payload["approach_total_motion_px"] = loop.approach_total_motion_px
        payload["approach_min_motion_px"] = loop.approach_min_motion_px
        payload["approach_prediction_error_px"] = loop.approach_prediction_error_px
        payload["approach_allowed_error_px"] = loop.approach_allowed_error_px
        payload["candidate_cell"] = loop.candidate_cell
        payload["projected_anchor"] = loop.projected_anchor
        payload["contact_recoveries"] = loop.contact_recoveries
        payload["contact_recovery_frame"] = loop.contact_recovery_frame
        payload["contact_recovery_cell"] = loop.contact_recovery_cell
        payload["external_cell_component_hits"] = (
            loop.external_cell_component_hits
        )
        payload["raw_z0_candidates"] = loop.raw_z0_candidates
        payload["scale_low_rejections"] = loop.scale_low_rejections
        payload["scale_high_rejections"] = loop.scale_high_rejections
        payload["scale_quantized_accepts"] = loop.scale_quantized_accepts
        payload["scale_diag_reason"] = loop.scale_diag_reason
        payload["scale_diag_cell"] = loop.scale_diag_cell
        payload["scale_diag_observed_px"] = loop.scale_diag_observed_px
        payload["scale_diag_expected_px"] = loop.scale_diag_expected_px
        payload["scale_diag_ratio"] = loop.scale_diag_ratio
        payload["z0_candidates"] = len(loop.z0_candidates)
        payload["up_confirmations"] = len(loop.up_confirmations)
        payload["z0_events"] = [
            {
                "frame_no": z.frame_no,
                "cell_id": z.cell_id,
                "region": z.region,
                "centroid_xy": list(z.centroid_xy),
                "floor_xy_bu": list(z.floor_xy_bu),
                "boundary_clearance_bu": z.boundary_clearance_bu,
                "scale_ratio": z.scale_ratio,
                "color_pixels": z.color_pixels,
            }
            for z in loop.z0_candidates
        ]
        payload["up_events"] = [
            {
                "contact_frame": u.contact_frame,
                "confirm_frame": u.confirm_frame,
                "cell_id": u.cell_id,
                "region": u.region,
                "contact_xy": list(u.contact_xy),
                "above_xy": list(u.above_xy),
                "up_bu": u.up_bu,
                "lateral_bu": u.lateral_bu,
                "scale_ratio_after": u.scale_ratio_after,
            }
            for u in loop.up_confirmations
        ]
    else:
        payload["changed_pixels"] = 0
        payload["ball_color_changed_pixels"] = 0
        payload["raw_ball_components"] = 0
        payload["ball_footprints"] = 0
        payload["merged_motion_footprints"] = 0
        payload["boundary_guard_rejections"] = 0
        payload["projected_signature_matches"] = 0
        payload["projected_signature_rejections"] = 0
        payload["approach_rejections"] = 0
        payload["motion_history_observations"] = 0
        payload["predicted_z0_cell"] = None
        payload["approach_reason"] = None
        payload["approach_direction_rejections"] = 0
        payload["approach_down_px_per_frame"] = None
        payload["approach_min_down_px_per_frame"] = None
        payload["trajectory_lock_state"] = None
        payload["trajectory_components_considered"] = 0
        payload["trajectory_topk_selected"] = 0
        payload["trajectory_candidate_rejections"] = 0
        payload["trajectory_predicted_xy"] = None
        payload["trajectory_gate_px"] = None
        payload["trajectory_lock_depth"] = 0
        payload["trajectory_topk_selected_xy"] = []
        payload["trajectory_bootstrap_straightness"] = None
        payload["approach_total_motion_px"] = 0.0
        payload["approach_min_motion_px"] = 0.0
        payload["approach_prediction_error_px"] = None
        payload["approach_allowed_error_px"] = None
        payload["candidate_cell"] = None
        payload["projected_anchor"] = None
        payload["contact_recoveries"] = 0
        payload["contact_recovery_frame"] = None
        payload["contact_recovery_cell"] = None
        payload["external_cell_component_hits"] = 0
        payload["raw_z0_candidates"] = 0
        payload["scale_low_rejections"] = 0
        payload["scale_high_rejections"] = 0
        payload["scale_quantized_accepts"] = 0
        payload["scale_diag_reason"] = None
        payload["scale_diag_cell"] = None
        payload["scale_diag_observed_px"] = None
        payload["scale_diag_expected_px"] = None
        payload["scale_diag_ratio"] = None
        payload["z0_candidates"] = 0
        payload["up_confirmations"] = 0
        payload["z0_events"] = []
        payload["up_events"] = []

    if include_image:
        ok, jpeg = cv2.imencode(
            ".jpg",
            result.rendered_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if ok:
            payload["rendered_jpeg_base64"] = base64.b64encode(
                jpeg.tobytes()
            ).decode("ascii")
        else:
            payload["rendered_jpeg_base64"] = None

    return payload

# CP-0036.1 is registered after the CP-0036 routes and shared registry exist.
from linecaller.api.live_wizard import register_wizard_routes

register_wizard_routes(app, registry, RUNTIME_UPLOAD_DIR)