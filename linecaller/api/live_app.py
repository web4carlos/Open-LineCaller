from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
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
    OfficialExternalLiveRuntime,
)


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNTIME_UPLOAD_DIR = ROOT / ".linecaller_runtime" / "live"


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
        payload["z0_candidates"] = len(
            result.frame_loop_result.z0_candidates
        )
        payload["up_confirmations"] = len(
            result.frame_loop_result.up_confirmations
        )
    else:
        payload["z0_candidates"] = 0
        payload["up_confirmations"] = 0

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