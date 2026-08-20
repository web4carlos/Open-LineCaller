from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from linecaller.api.boundary_calibration import (
    infer_court_corners_from_boundary_lines,
    parse_boundary_lines_json,
)
from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibrator,
    ExternalGridConfig,
)


@dataclass
class WizardState:
    calibration_path: Path | None = None
    background_path: Path | None = None
    ball_template_path: Path | None = None
    image_size: tuple[int, int] | None = None
    coverage: str | None = None
    external_cells: int = 0
    image_points: tuple[tuple[float, float], ...] = ()
    calibration_method: str | None = None
    offscreen_corners: int = 0

    def reset(self) -> None:
        self.calibration_path = None
        self.background_path = None
        self.ball_template_path = None
        self.image_size = None
        self.coverage = None
        self.external_cells = 0
        self.image_points = ()
        self.calibration_method = None
        self.offscreen_corners = 0


wizard_state = WizardState()


def _decode_upload(upload: UploadFile, label: str) -> tuple[np.ndarray, Image.Image]:
    data = upload.file.read()
    if not data:
        raise ValueError(f"{label} is empty")
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{label} is not a valid image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, Image.fromarray(rgb, mode="RGB")


def _parse_image_points(raw: str, image_size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("image_points must be valid JSON") from exc

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            "Select exactly 4 court corners: near-left, near-right, far-right, far-left"
        )

    points: list[tuple[float, float]] = []
    width, height = image_size
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each court corner must contain x,y")
        x, y = float(item[0]), float(item[1])
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError("A selected court corner is outside the captured frame")
        points.append((x, y))

    polygon = np.asarray(points, dtype=np.float32)
    if abs(float(cv2.contourArea(polygon))) < 250.0:
        raise ValueError("Selected court quadrilateral is too small")
    if not cv2.isContourConvex(np.rint(polygon).astype(np.int32)):
        raise ValueError(
            "Court corners cross each other. Use: near-left -> near-right -> far-right -> far-left"
        )
    return tuple(points)


def _parse_interactive_quad(
    raw: str,
    image_size: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...], int]:
    # Interactive handles may represent a mathematical court corner that is
    # outside the camera image. The legacy visible-corner mode remains strict.
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("quad_points must be valid JSON") from exc

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            "quad_points must contain 4 corners: "
            "near-left, near-right, far-right, far-left"
        )

    width, height = image_size
    diagonal = math.hypot(width, height)
    max_extension = 1.5 * diagonal
    points: list[tuple[float, float]] = []
    offscreen = 0

    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"quad corner {index + 1} must contain x,y")
        x, y = float(item[0]), float(item[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"quad corner {index + 1} is not finite")

        dx = 0.0 if 0.0 <= x <= width else (-x if x < 0.0 else x - width)
        dy = 0.0 if 0.0 <= y <= height else (-y if y < 0.0 else y - height)
        if math.hypot(dx, dy) > max_extension:
            raise ValueError(
                f"quad corner {index + 1} is implausibly far outside the frame"
            )
        if not (0.0 <= x < width and 0.0 <= y < height):
            offscreen += 1
        points.append((x, y))

    polygon = np.asarray(points, dtype=np.float32)
    if abs(float(cv2.contourArea(polygon))) < max(250.0, 0.002 * width * height):
        raise ValueError("Interactive court quadrilateral is too small")
    if not cv2.isContourConvex(np.rint(polygon).astype(np.int32)):
        raise ValueError(
            "Interactive court corners cross. Keep order: "
            "near-left -> near-right -> far-right -> far-left"
        )

    for i in range(4):
        if math.dist(points[i], points[(i + 1) % 4]) < 8.0:
            raise ValueError("Two adjacent court handles are too close together")

    return tuple(points), offscreen


def _summary_dict(registry: Any) -> dict[str, Any]:
    registry.sync_runtime_summary()
    summary = registry.summary
    return {
        "configured": bool(summary.configured),
        "camera_id": summary.camera_id,
        "mount_position": summary.mount_position,
        "zones": list(summary.zones),
        "receiving_side": summary.receiving_side,
        "image_size": list(summary.image_size) if summary.image_size else None,
        "active_external_cells": summary.active_external_cells,
        "pose_epoch": summary.pose_epoch,
        "wizard": {
            "calibrated": wizard_state.calibration_path is not None,
            "ball_selected": wizard_state.ball_template_path is not None,
            "coverage": wizard_state.coverage,
            "external_cells": wizard_state.external_cells,
            "image_points": [list(p) for p in wizard_state.image_points],
            "calibration_method": wizard_state.calibration_method,
            "offscreen_corners": wizard_state.offscreen_corners,
        },
    }


def register_wizard_routes(app: Any, registry: Any, runtime_upload_dir: Path) -> None:
    router = APIRouter(prefix="/api/wizard", tags=["live-wizard"])
    wizard_dir = Path(runtime_upload_dir) / "wizard"

    @router.get("/status")
    def status() -> dict[str, Any]:
        return _summary_dict(registry)

    @router.post("/reset")
    def reset() -> dict[str, Any]:
        wizard_state.reset()
        registry.runtime = None
        registry._frame_no = 0
        registry.summary.configured = False
        registry.summary.camera_id = None
        registry.summary.mount_position = None
        registry.summary.zones = ()
        registry.summary.receiving_side = None
        registry.summary.calibration_path = None
        registry.summary.background_path = None
        registry.summary.ball_template_path = None
        registry.summary.image_size = None
        registry.summary.active_external_cells = 0
        registry.summary.pose_epoch = 0
        return _summary_dict(registry)

    @router.post("/calibrate")
    def calibrate(
        background: UploadFile = File(...),
        image_points: str | None = Form(None),
        boundary_lines: str | None = Form(None),
        quad_points: str | None = Form(None),
        coverage: str = Form("FULL_COURT"),
        margin_bu: float = Form(10.0),
    ) -> dict[str, Any]:
        try:
            bgr, rgb_image = _decode_upload(background, "background")
            height, width = bgr.shape[:2]
            size = (int(width), int(height))

            if quad_points:
                points, offscreen_corners = _parse_interactive_quad(
                    quad_points,
                    size,
                )
                calibration_method = "INTERACTIVE_QUAD"
            elif boundary_lines:
                parsed_lines = parse_boundary_lines_json(boundary_lines, size)
                inference = infer_court_corners_from_boundary_lines(parsed_lines, size)
                points = inference.image_points
                calibration_method = "BOUNDARY_LINES"
                offscreen_corners = inference.offscreen_corners
            elif image_points:
                points = _parse_image_points(image_points, size)
                calibration_method = "VISIBLE_CORNERS"
                offscreen_corners = 0
            else:
                raise ValueError(
                    "Provide quad_points, image_points or boundary_lines "
                    "calibration evidence"
                )

            mode = CalibrationCoverage.parse(coverage)

            if float(margin_bu) <= 0.0:
                raise ValueError("margin_bu must be > 0")

            wizard_dir.mkdir(parents=True, exist_ok=True)
            background_path = wizard_dir / "EXTERNAL_GRID_BACKGROUND.png"
            calibration_path = wizard_dir / "EXTERNAL_GRID_CALIBRATION.json"

            rgb_image.save(background_path, format="PNG")

            builder = ExternalGridCalibrator(
                points,
                size,
                coverage=mode,
                config=ExternalGridConfig(margin_bu=float(margin_bu)),
                image_up_unit=(0.0, -1.0),
            )
            calibration = builder.build(
                background_image=background_path.name
            )
            if not calibration.cells:
                raise ValueError(
                    "No visible external cells were produced by this calibration"
                )
            calibration.save(calibration_path)

            wizard_state.calibration_path = calibration_path.resolve()
            wizard_state.background_path = background_path.resolve()
            wizard_state.ball_template_path = None
            wizard_state.image_size = size
            wizard_state.coverage = mode.value
            wizard_state.external_cells = len(calibration.cells)
            wizard_state.image_points = points
            wizard_state.calibration_method = calibration_method
            wizard_state.offscreen_corners = offscreen_corners

            # Court geometry changed: old runtime must not continue.
            registry.runtime = None
            registry.summary.configured = False
            registry.summary.active_external_cells = 0

            return {
                "ok": True,
                "coverage": mode.value,
                "image_size": list(size),
                "external_cells": len(calibration.cells),
                "image_points": [list(p) for p in points],
                "calibration_method": calibration_method,
                "offscreen_corners": offscreen_corners,
                "next": "SELECT_BALL",
            }
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/select-ball")
    def select_ball(
        frame: UploadFile = File(...),
        x: float = Form(...),
        y: float = Form(...),
        crop_size: int = Form(36),
        camera_id: str = Form("net-camera-01"),
        mount_position: str = Form("NET_CENTER"),
        zones: str = Form(
            "FAR_LEFT,FAR_BASELINE,FAR_RIGHT,"
            "NEAR_LEFT,NEAR_BASELINE,NEAR_RIGHT"
        ),
        receiving_side: str = Form("FAR"),
        depth_bu: float = Form(6.0),
    ) -> dict[str, Any]:
        if wizard_state.calibration_path is None or wizard_state.background_path is None:
            raise HTTPException(
                status_code=409,
                detail="Calibrate the court before selecting the ball",
            )

        try:
            _, rgb_image = _decode_upload(frame, "ball frame")
            if wizard_state.image_size is None:
                raise ValueError("Wizard calibration image size is missing")
            if rgb_image.size != wizard_state.image_size:
                raise ValueError(
                    "Ball-selection frame size does not match the court calibration"
                )

            width, height = rgb_image.size
            cx, cy = int(round(float(x))), int(round(float(y)))
            if not (0 <= cx < width and 0 <= cy < height):
                raise ValueError("Selected ball center is outside the frame")

            size = max(16, min(96, int(crop_size)))
            half = size // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(width, cx + half)
            y1 = min(height, cy + half)
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise ValueError("Ball selection is too close to the frame edge")

            wizard_dir.mkdir(parents=True, exist_ok=True)
            ball_path = wizard_dir / "BALL_TEMPLATE.png"
            rgb_image.crop((x0, y0, x1, y1)).save(
                ball_path,
                format="PNG",
            )
            wizard_state.ball_template_path = ball_path.resolve()

            zone_list = [
                token.strip().upper()
                for token in zones.split(",")
                if token.strip()
            ]
            if not zone_list:
                raise ValueError("Select at least one camera-owned external zone")

            registry.configure_paths(
                calibration_path=wizard_state.calibration_path,
                ball_template_path=wizard_state.ball_template_path,
                background_path=wizard_state.background_path,
                camera_id=camera_id,
                mount_position=mount_position,
                zones=zone_list,
                receiving_side=receiving_side,
                depth_bu=float(depth_bu),
            )
            result = _summary_dict(registry)
            result["ok"] = True
            result["next"] = "READY"
            result["ball_crop"] = [x0, y0, x1, y1]
            return result
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.include_router(router)