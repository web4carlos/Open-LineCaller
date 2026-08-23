from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibration,
    ExternalGridCalibrator,
    ExternalGridConfig,
)
from linecaller.dcf.predicted_topk_selector import PredictedTopKSelector


FEATURE_VERSION = "CP-0036.2.7"


@dataclass(frozen=True)
class SyntheticHalfCourtConfig:
    image_size: tuple[int, int] = (960, 540)
    fps: int = 30
    image_points: tuple[tuple[float, float], ...] = (
        (80.0, 500.0),
        (880.0, 500.0),
        (700.0, 120.0),
        (260.0, 120.0),
    )
    court_rgb: tuple[int, int, int] = (52, 112, 78)
    outside_rgb: tuple[int, int, int] = (42, 47, 52)
    line_rgb: tuple[int, int, int] = (242, 242, 238)
    ball_rgb: tuple[int, int, int] = (218, 255, 34)


@dataclass(frozen=True)
class SyntheticFrameTruth:
    frame_no: int
    scenario: str
    ball_floor_xy_bu: tuple[float, float] | None = None
    ball_height_px: float = 0.0
    distractor_floor_xy_bu: tuple[float, float] | None = None
    contact: bool = False
    expected_call: str | None = None


@dataclass(frozen=True)
class SyntheticReferenceBundle:
    output_dir: str
    background: str
    calibration: str
    ball_template: str
    truth: str
    video: str | None
    frames_dir: str | None
    frame_count: int
    fps: int


class PlayerNetHalfSyntheticReference:
    """Deterministic PLAYER NET HALF engineering reference."""

    def __init__(
        self,
        config: SyntheticHalfCourtConfig | None = None,
    ) -> None:
        self.config = config or SyntheticHalfCourtConfig()
        self._calibrator = ExternalGridCalibrator(
            self.config.image_points,
            self.config.image_size,
            coverage="HALF_COURT",
            config=ExternalGridConfig(margin_bu=10.0),
            image_up_unit=(0.0, -1.0),
        )
        self.calibration = self._calibrator.build(
            background_image="background.png"
        )

    def project_floor_xy(
        self,
        floor_xy_bu: tuple[float, float],
    ) -> tuple[float, float]:
        return self._calibrator.project_top_view(*floor_xy_bu)

    def expected_ball_diameter_px(
        self,
        floor_xy_bu: tuple[float, float],
    ) -> float:
        x, y = floor_xy_bu
        a = self.project_floor_xy((x - 0.5, y))
        b = self.project_floor_xy((x + 0.5, y))
        return max(2.0, float(math.dist(a, b)))

    def background_rgb(self) -> np.ndarray:
        width, height = self.config.image_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = self.config.outside_rgb
        pts = np.asarray(self.config.image_points, dtype=np.int32)
        cv2.fillConvexPoly(
            image, pts, self.config.court_rgb, lineType=cv2.LINE_AA
        )
        p = self.config.image_points
        for a, b in ((p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0])):
            cv2.line(
                image,
                tuple(round(v) for v in a),
                tuple(round(v) for v in b),
                self.config.line_rgb,
                3,
                cv2.LINE_AA,
            )

        # Visual kitchen/service lines only. They are not an operational grid.
        a = self.project_floor_xy((0.0, 33.0))
        b = self.project_floor_xy((83.0, 33.0))
        cv2.line(
            image,
            tuple(round(v) for v in a),
            tuple(round(v) for v in b),
            self.config.line_rgb,
            2,
            cv2.LINE_AA,
        )
        a = self.project_floor_xy((41.5, 0.0))
        b = self.project_floor_xy((41.5, 33.0))
        cv2.line(
            image,
            tuple(round(v) for v in a),
            tuple(round(v) for v in b),
            self.config.line_rgb,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            tuple(round(v) for v in p[0]),
            tuple(round(v) for v in p[1]),
            (28, 30, 32),
            5,
            cv2.LINE_AA,
        )
        return image

    def ball_template_rgb(self) -> np.ndarray:
        image = Image.new("RGB", (48, 48), (45, 48, 50))
        draw = ImageDraw.Draw(image)
        draw.ellipse((6, 6, 41, 41), fill=self.config.ball_rgb)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return float(a + (b - a) * t)

    def truth_frames(self) -> tuple[SyntheticFrameTruth, ...]:
        frames: list[SyntheticFrameTruth] = []

        for frame_no in range(9):
            frames.append(
                SyntheticFrameTruth(
                    frame_no,
                    "DEEP_DISTRACTOR_ONLY",
                    distractor_floor_xy_bu=(
                        68.0 - 0.55 * frame_no,
                        43.0 - 0.45 * frame_no,
                    ),
                )
            )

        # Real ball enters at net y~90 and contacts OUT_LEFT at frame 24.
        for t in range(20):
            frame_no = 9 + t
            if t <= 15:
                u = t / 15.0
                x = self._lerp(42.0, -2.2, u)
                y = self._lerp(90.0, 18.9, u)
                h = self._lerp(26.0, 0.0, u)
            else:
                u = (t - 15) / 4.0
                x = self._lerp(-2.2, -6.0, u)
                y = self._lerp(18.9, 10.0, u)
                h = self._lerp(0.0, 24.0, u)
            frames.append(
                SyntheticFrameTruth(
                    frame_no,
                    "INGRESS_OUT_LEFT",
                    ball_floor_xy_bu=(x, y),
                    ball_height_px=h,
                    distractor_floor_xy_bu=(
                        63.0 - 0.20 * t,
                        38.0 - 0.10 * t,
                    ),
                    contact=t == 15,
                    expected_call="OUT_LEFT" if t == 15 else None,
                )
            )

        for frame_no in range(29, 37):
            frames.append(SyntheticFrameTruth(frame_no, "RESET_GAP"))

        # Second independent ingress; contact is safely IN at frame 52.
        for t in range(21):
            frame_no = 37 + t
            if t <= 15:
                u = t / 15.0
                x = self._lerp(56.0, 31.0, u)
                y = self._lerp(90.0, 29.0, u)
                h = self._lerp(24.0, 0.0, u)
            else:
                u = (t - 15) / 5.0
                x = self._lerp(31.0, 27.0, u)
                y = self._lerp(29.0, 20.0, u)
                h = self._lerp(0.0, 25.0, u)
            frames.append(
                SyntheticFrameTruth(
                    frame_no,
                    "INGRESS_IN_COURT",
                    ball_floor_xy_bu=(x, y),
                    ball_height_px=h,
                    contact=t == 15,
                    expected_call="IN" if t == 15 else None,
                )
            )

        for frame_no in range(58, 65):
            frames.append(SyntheticFrameTruth(frame_no, "RESET_GAP"))

        for frame_no in range(65, 75):
            t = frame_no - 65
            frames.append(
                SyntheticFrameTruth(
                    frame_no,
                    "DEEP_DISTRACTOR_REACQUIRE",
                    distractor_floor_xy_bu=(
                        22.0 + 0.45 * t,
                        47.0 - 0.55 * t,
                    ),
                )
            )
        return tuple(frames)

    def _draw_object(
        self,
        image: np.ndarray,
        floor_xy_bu: tuple[float, float],
        *,
        height_px: float,
        square: bool,
    ) -> None:
        cx, cy = self.project_floor_xy(floor_xy_bu)
        cy -= float(height_px)
        radius = max(
            2,
            int(round(0.5 * self.expected_ball_diameter_px(floor_xy_bu))),
        )
        center = (int(round(cx)), int(round(cy)))
        if square:
            cv2.rectangle(
                image,
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
                self.config.ball_rgb,
                -1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(
                image,
                center,
                radius,
                self.config.ball_rgb,
                -1,
                cv2.LINE_AA,
            )

    def render_frame(self, truth: SyntheticFrameTruth) -> np.ndarray:
        image = self.background_rgb().copy()
        if truth.distractor_floor_xy_bu is not None:
            self._draw_object(
                image,
                truth.distractor_floor_xy_bu,
                height_px=0.0,
                square=True,
            )
        if truth.ball_floor_xy_bu is not None:
            self._draw_object(
                image,
                truth.ball_floor_xy_bu,
                height_px=truth.ball_height_px,
                square=False,
            )
        return image

    def manifest(self) -> dict[str, Any]:
        truth = self.truth_frames()
        return {
            "feature_version": FEATURE_VERSION,
            "product_mode": "PLAYER NET HALF",
            "coverage": "HALF_COURT",
            "image_size": list(self.config.image_size),
            "fps": self.config.fps,
            "coordinate_convention": {
                "x_bu": "0..83 between sidelines",
                "y_bu": "0 baseline, 91 net edge",
                "outside_grid_only": True,
            },
            "contacts": [asdict(f) for f in truth if f.contact],
            "frames": [asdict(f) for f in truth],
            "note": (
                "Deterministic engineering reference only; real phone-on-net "
                "footage remains the physical validation gate."
            ),
        }

    @staticmethod
    def _write_video(
        stem: Path,
        frames_rgb: list[np.ndarray],
        fps: int,
        image_size: tuple[int, int],
    ) -> Path | None:
        width, height = image_size
        for path, codec in (
            (stem.with_suffix(".mp4"), "mp4v"),
            (stem.with_suffix(".avi"), "MJPG"),
        ):
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                continue
            for frame in frames_rgb:
                writer.write(frame[:, :, ::-1])
            writer.release()
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def build_bundle(
        self,
        output_dir: str | Path,
        *,
        write_video: bool = True,
        write_frames: bool = True,
    ) -> SyntheticReferenceBundle:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        background_path = root / "background.png"
        calibration_path = root / "calibration_half_court.json"
        template_path = root / "ball_template.png"
        truth_path = root / "truth.json"

        Image.fromarray(self.background_rgb(), mode="RGB").save(background_path)
        Image.fromarray(self.ball_template_rgb(), mode="RGB").save(template_path)

        calibration = ExternalGridCalibration(
            version=self.calibration.version,
            coverage=self.calibration.coverage,
            image_size=self.calibration.image_size,
            image_points=self.calibration.image_points,
            config=self.calibration.config,
            background_image=str(background_path),
            image_up_unit=self.calibration.image_up_unit,
            cells=self.calibration.cells,
        )
        calibration.save(calibration_path)

        frames_dir = root / "frames" if write_frames else None
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)

        video_frames: list[np.ndarray] = []
        truth = self.truth_frames()
        for item in truth:
            frame = self.render_frame(item)
            if write_video:
                video_frames.append(frame)
            if frames_dir is not None:
                Image.fromarray(frame, mode="RGB").save(
                    frames_dir / f"frame_{item.frame_no:04d}.png"
                )

        video_path = (
            self._write_video(
                root / "player_net_half_reference",
                video_frames,
                self.config.fps,
                self.config.image_size,
            )
            if write_video
            else None
        )

        manifest = self.manifest()
        manifest["files"] = {
            "background": background_path.name,
            "calibration": calibration_path.name,
            "ball_template": template_path.name,
            "video": None if video_path is None else video_path.name,
            "frames_dir": None if frames_dir is None else frames_dir.name,
        }
        truth_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (root / "README.txt").write_text(
            "Open LineCaller PLAYER NET HALF synthetic reference\n"
            f"Feature: {FEATURE_VERSION}\n"
            "frame 24 -> OUT_LEFT\n"
            "frame 52 -> IN\n"
            "Synthetic engineering evidence only; real footage is still required.\n",
            encoding="utf-8",
        )

        return SyntheticReferenceBundle(
            output_dir=str(root),
            background=str(background_path),
            calibration=str(calibration_path),
            ball_template=str(template_path),
            truth=str(truth_path),
            video=None if video_path is None else str(video_path),
            frames_dir=None if frames_dir is None else str(frames_dir),
            frame_count=len(truth),
            fps=self.config.fps,
        )


def _component(
    scene: PlayerNetHalfSyntheticReference,
    floor_xy_bu: tuple[float, float],
) -> BallComponent:
    x, y = scene.project_floor_xy(floor_xy_bu)
    return BallComponent(
        bbox=(int(round(x)) - 2, int(round(y)) - 2, 5, 5),
        area=9,
        centroid_xy=(x, y),
        scale_px=scene.expected_ball_diameter_px(floor_xy_bu),
    )


def run_net_ingress_selector_gate() -> dict[str, Any]:
    scene = PlayerNetHalfSyntheticReference()
    selector = PredictedTopKSelector(
        calibration=scene.calibration,
        top_k=3,
        history_frames=3,
        min_prior_observations=2,
        min_gate_px=1000.0,
        bootstrap_min_motion_px=0.5,
        bootstrap_min_straightness=0.50,
        bootstrap_receiving_side_guard=True,
        receiving_side="FAR",
        bootstrap_net_margin_bu=4.0,
        bootstrap_net_ingress_guard=True,
        bootstrap_ingress_depth_bu=36.0,
        bootstrap_min_inward_bu=1.0,
        max_misses=2,
    )

    deep = tuple(
        _component(scene, p)
        for p in ((68.0, 43.0), (67.0, 42.0), (66.0, 41.0))
    )
    deep_result = selector.select(
        3,
        [deep[2]],
        [(1, (deep[0],)), (2, (deep[1],))],
    )

    d = tuple(
        _component(scene, p)
        for p in ((63.0, 38.0), (62.5, 37.5), (62.0, 37.0))
    )
    r = tuple(
        _component(scene, p)
        for p in ((42.0, 89.0), (39.0, 82.0), (36.0, 75.0))
    )
    real_result = selector.select(
        6,
        [d[2], r[2]],
        [(4, (d[0], r[0])), (5, (d[1], r[1]))],
    )

    last = selector.last_component
    lock_error = (
        math.inf
        if last is None
        else math.dist(last.centroid_xy, r[2].centroid_xy)
    )
    passed = (
        deep_result.state == "UNLOCKED"
        and deep_result.bootstrap_ingress_rejections > 0
        and real_result.state == "BOOTSTRAP"
        and real_result.bootstrap_ingress_reason == "ACCEPTED"
        and lock_error < 1e-6
    )
    return {
        "feature_version": FEATURE_VERSION,
        "passed": bool(passed),
        "deep_state": deep_result.state,
        "deep_ingress_rejections": deep_result.bootstrap_ingress_rejections,
        "deep_reason": deep_result.bootstrap_ingress_reason,
        "real_state": real_result.state,
        "real_reason": real_result.bootstrap_ingress_reason,
        "real_inward_bu": real_result.bootstrap_ingress_delta_bu,
        "lock_error_px": float(lock_error),
    }
