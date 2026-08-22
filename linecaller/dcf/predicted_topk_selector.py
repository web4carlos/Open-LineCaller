from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalGridCalibration,
    rasterized_scale_compatible,
)


@dataclass(frozen=True)
class TrajectoryPrediction:
    predicted_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    speed_px_per_frame: float
    last_component: BallComponent
    history_depth: int


class PerspectiveBallScaleModel:
    """Continuous homography scale evidence; no interior grid traversal."""

    def __init__(self, calibration: ExternalGridCalibration) -> None:
        self.calibration = calibration
        cfg = calibration.config
        court_y = cfg.court_y_bu(
            __import__(
                "linecaller.dcf.external_grid_frame_loop",
                fromlist=["CalibrationCoverage"],
            ).CalibrationCoverage.parse(calibration.coverage)
        )
        court = np.asarray(
            (
                (0.0, court_y),
                (cfg.court_x_bu, court_y),
                (cfg.court_x_bu, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        )
        image = np.asarray(
            calibration.image_points,
            dtype=np.float32,
        )
        self._top_to_image = cv2.getPerspectiveTransform(
            court,
            image,
        )
        self._image_to_top = cv2.getPerspectiveTransform(
            image,
            court,
        )

    def expected_diameter_px(
        self,
        image_xy: tuple[float, float],
    ) -> float | None:
        p = np.asarray(
            [[[float(image_xy[0]), float(image_xy[1])]]],
            dtype=np.float32,
        )
        floor = cv2.perspectiveTransform(
            p,
            self._image_to_top,
        )[0, 0]
        x_bu = float(floor[0])
        y_bu = float(floor[1])
        if not (math.isfinite(x_bu) and math.isfinite(y_bu)):
            return None

        lateral = np.asarray(
            [[
                [x_bu - 0.5, y_bu],
                [x_bu + 0.5, y_bu],
            ]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(
            lateral,
            self._top_to_image,
        )[0]
        expected = float(
            math.dist(
                tuple(float(v) for v in projected[0]),
                tuple(float(v) for v in projected[1]),
            )
        )
        if not math.isfinite(expected) or expected <= 0.0:
            return None
        return max(0.5, expected)


@dataclass(frozen=True)
class PredictedTopKSelection:
    state: str
    predicted_xy: tuple[float, float] | None
    gate_px: float | None
    considered: int
    selected: tuple[BallComponent, ...]
    rejected_by_distance: int
    history_depth: int
    bootstrap_straightness: float | None = None
    bootstrap_scale_rejections: int = 0
    bootstrap_scale_observed_px: float | None = None
    bootstrap_scale_expected_px: float | None = None
    bootstrap_scale_ratio: float | None = None


class PredictedTopKSelector:
    """Single-trajectory image-space lock + nearest-K component selector.

    This selector never decides IN/OUT and never traverses an interior court
    grid. It only limits which image components may continue the locked ball
    trajectory before the existing outside-grid Z0 safety gates run.
    """

    def __init__(
        self,
        *,
        top_k: int = 3,
        history_frames: int = 3,
        min_prior_observations: int = 2,
        min_gate_px: float = 5.0,
        gate_scale: float = 4.75,
        bootstrap_min_motion_px: float = 1.5,
        bootstrap_min_straightness: float = 0.70,
        calibration: ExternalGridCalibration | None = None,
        bootstrap_scale_guard: bool = False,
        bootstrap_min_scale_ratio: float = 0.35,
        bootstrap_max_scale_ratio: float = 2.20,
        max_misses: int = 2,
    ) -> None:
        self.top_k = int(top_k)
        self.history_frames = int(history_frames)
        self.min_prior_observations = int(min_prior_observations)
        self.min_gate_px = float(min_gate_px)
        self.gate_scale = float(gate_scale)
        self.bootstrap_min_motion_px = float(bootstrap_min_motion_px)
        self.bootstrap_min_straightness = float(
            bootstrap_min_straightness
        )
        self.bootstrap_scale_guard = bool(bootstrap_scale_guard)
        self.bootstrap_min_scale_ratio = float(
            bootstrap_min_scale_ratio
        )
        self.bootstrap_max_scale_ratio = float(
            bootstrap_max_scale_ratio
        )
        self._perspective_scale = (
            PerspectiveBallScaleModel(calibration)
            if calibration is not None
            else None
        )
        self.max_misses = int(max_misses)

        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.history_frames < 2:
            raise ValueError("history_frames must be >= 2")
        if not (
            1 <= self.min_prior_observations <= self.history_frames
        ):
            raise ValueError(
                "min_prior_observations must be in [1, history_frames]"
            )
        if self.min_gate_px <= 0.0:
            raise ValueError("min_gate_px must be > 0")
        if self.gate_scale <= 0.0:
            raise ValueError("gate_scale must be > 0")
        if self.bootstrap_min_motion_px <= 0.0:
            raise ValueError("bootstrap_min_motion_px must be > 0")
        if not 0.0 < self.bootstrap_min_straightness <= 1.0:
            raise ValueError(
                "bootstrap_min_straightness must be in (0,1]"
            )
        if self.bootstrap_min_scale_ratio <= 0.0:
            raise ValueError(
                "bootstrap_min_scale_ratio must be > 0"
            )
        if (
            self.bootstrap_max_scale_ratio
            < self.bootstrap_min_scale_ratio
        ):
            raise ValueError(
                "bootstrap_max_scale_ratio must be >= "
                "bootstrap_min_scale_ratio"
            )
        if self.bootstrap_scale_guard and self._perspective_scale is None:
            raise ValueError(
                "bootstrap_scale_guard requires calibration"
            )
        if self.max_misses < 0:
            raise ValueError("max_misses must be >= 0")

        self._history: deque[tuple[int, BallComponent]] = deque(
            maxlen=self.history_frames + 1
        )
        self._misses = 0
        self._reset_bootstrap_scale_telemetry()

    @property
    def locked(self) -> bool:
        return bool(self._history)

    @property
    def history_depth(self) -> int:
        return len(self._history)

    @property
    def last_component(self) -> BallComponent | None:
        return self._history[-1][1] if self._history else None

    def reset(self) -> None:
        self._history.clear()
        self._misses = 0

    def _reset_bootstrap_scale_telemetry(self) -> None:
        self._bootstrap_scale_rejections = 0
        self._bootstrap_scale_observed_px: float | None = None
        self._bootstrap_scale_expected_px: float | None = None
        self._bootstrap_scale_ratio: float | None = None

    def _bootstrap_scale_ok(
        self,
        component: BallComponent,
    ) -> bool:
        if not self.bootstrap_scale_guard:
            return True
        if self._perspective_scale is None:
            return False

        expected = self._perspective_scale.expected_diameter_px(
            component.centroid_xy
        )
        if expected is None:
            self._bootstrap_scale_rejections += 1
            self._bootstrap_scale_observed_px = float(
                component.scale_px
            )
            self._bootstrap_scale_expected_px = None
            self._bootstrap_scale_ratio = None
            return False

        observed = float(component.scale_px)
        ratio = observed / max(0.5, float(expected))
        compatible = rasterized_scale_compatible(
            observed,
            expected,
            self.bootstrap_min_scale_ratio,
            self.bootstrap_max_scale_ratio,
        )
        if not compatible:
            self._bootstrap_scale_rejections += 1
            self._bootstrap_scale_observed_px = observed
            self._bootstrap_scale_expected_px = float(expected)
            self._bootstrap_scale_ratio = float(ratio)
        return bool(compatible)

    @staticmethod
    def _compatible_scale(
        a: BallComponent,
        b: BallComponent,
    ) -> bool:
        lo = min(float(a.scale_px), float(b.scale_px))
        hi = max(float(a.scale_px), float(b.scale_px))
        if lo <= 0.0:
            return False
        return (hi / lo) <= 2.60

    def _gate_px(
        self,
        a: BallComponent,
        b: BallComponent | None = None,
    ) -> float:
        scale = float(a.scale_px)
        if b is not None:
            scale = max(scale, float(b.scale_px))
        return max(self.min_gate_px, self.gate_scale * scale)

    @staticmethod
    def _linear_prediction(
        history: list[tuple[int, BallComponent]],
        frame_no: int,
    ) -> tuple[tuple[float, float] | None, tuple[float, float]]:
        if len(history) < 2:
            return None, (0.0, 0.0)

        t = np.asarray(
            [float(f) for f, _ in history],
            dtype=np.float64,
        )
        xs = np.asarray(
            [float(c.centroid_xy[0]) for _, c in history],
            dtype=np.float64,
        )
        ys = np.asarray(
            [float(c.centroid_xy[1]) for _, c in history],
            dtype=np.float64,
        )
        tt = t - t.mean()
        denom = float(np.dot(tt, tt))
        if denom <= 1e-9:
            return None, (0.0, 0.0)

        vx = float(np.dot(tt, xs - xs.mean()) / denom)
        vy = float(np.dot(tt, ys - ys.mean()) / denom)
        dt = float(frame_no) - float(t.mean())
        predicted = (
            float(xs.mean() + vx * dt),
            float(ys.mean() + vy * dt),
        )
        return predicted, (vx, vy)

    @staticmethod
    def _straightness(
        chain: list[tuple[int, BallComponent]],
    ) -> float:
        if len(chain) < 2:
            return 0.0
        points = [c.centroid_xy for _, c in chain]
        path = sum(
            math.dist(points[i - 1], points[i])
            for i in range(1, len(points))
        )
        if path <= 1e-9:
            return 0.0
        return float(math.dist(points[0], points[-1]) / path)

    @staticmethod
    def _fit_rms(
        chain: list[tuple[int, BallComponent]],
    ) -> float:
        if len(chain) < 2:
            return math.inf
        t = np.asarray(
            [float(f) for f, _ in chain],
            dtype=np.float64,
        )
        xs = np.asarray(
            [float(c.centroid_xy[0]) for _, c in chain],
            dtype=np.float64,
        )
        ys = np.asarray(
            [float(c.centroid_xy[1]) for _, c in chain],
            dtype=np.float64,
        )
        tt = t - t.mean()
        denom = float(np.dot(tt, tt))
        if denom <= 1e-9:
            return math.inf
        vx = float(np.dot(tt, xs - xs.mean()) / denom)
        vy = float(np.dot(tt, ys - ys.mean()) / denom)
        fit_x = xs.mean() + vx * tt
        fit_y = ys.mean() + vy * tt
        residual2 = (xs - fit_x) ** 2 + (ys - fit_y) ** 2
        return float(math.sqrt(float(np.mean(residual2))))

    def _strict_prior_chain(
        self,
        frame_no: int,
        current: BallComponent,
        motion_frames: Iterable[
            tuple[int, tuple[BallComponent, ...]]
        ],
    ) -> list[tuple[int, BallComponent]]:
        chain_reverse: list[tuple[int, BallComponent]] = []
        target = current

        frames = [
            (int(f), tuple(components))
            for f, components in motion_frames
            if int(f) < int(frame_no)
            and int(frame_no) - int(f) <= self.history_frames + 1
        ]

        for prior_frame, components in reversed(frames):
            strict = [
                component
                for component in components
                if not component.recovered
                and self._bootstrap_scale_ok(component)
                and self._compatible_scale(target, component)
            ]
            if not strict:
                continue

            ranked = sorted(
                (
                    (
                        math.dist(
                            target.centroid_xy,
                            component.centroid_xy,
                        ),
                        component,
                    )
                    for component in strict
                ),
                key=lambda item: item[0],
            )
            distance, chosen = ranked[0]
            if distance > self._gate_px(target, chosen):
                continue

            chain_reverse.append((prior_frame, chosen))
            target = chosen
            if len(chain_reverse) >= self.history_frames:
                break

        return list(reversed(chain_reverse))

    def _bootstrap(
        self,
        frame_no: int,
        components: list[BallComponent],
        motion_frames: Iterable[
            tuple[int, tuple[BallComponent, ...]]
        ],
    ) -> tuple[float | None, tuple[float, float] | None]:
        options = []

        for current in components:
            if current.recovered:
                continue
            if not self._bootstrap_scale_ok(current):
                continue

            prior = self._strict_prior_chain(
                frame_no,
                current,
                motion_frames,
            )
            if len(prior) < self.min_prior_observations:
                continue

            predicted, _ = self._linear_prediction(
                prior,
                frame_no,
            )
            if predicted is None:
                continue

            error = float(
                math.dist(predicted, current.centroid_xy)
            )
            gate = self._gate_px(prior[-1][1], current)
            if error > gate:
                continue

            chain = [*prior, (int(frame_no), current)]
            total_motion = float(
                math.dist(
                    chain[0][1].centroid_xy,
                    current.centroid_xy,
                )
            )
            if total_motion < self.bootstrap_min_motion_px:
                continue

            straightness = self._straightness(chain)
            if straightness < self.bootstrap_min_straightness:
                continue

            rms = self._fit_rms(chain)
            scale = max(0.5, float(current.scale_px))
            score = (
                -float(len(chain)),
                rms / scale,
                error / scale,
                -total_motion / scale,
            )
            options.append(
                (
                    score,
                    chain,
                    straightness,
                    predicted,
                )
            )

        if not options:
            return None, None

        _, chain, straightness, predicted = min(
            options,
            key=lambda item: item[0],
        )
        self._history.clear()
        self._history.extend(
            chain[-(self.history_frames + 1):]
        )
        self._misses = 0
        return float(straightness), predicted

    def prediction(
        self,
        frame_no: int,
    ) -> TrajectoryPrediction | None:
        if not self._history:
            return None
        history = list(self._history)
        predicted, velocity = self._linear_prediction(
            history,
            int(frame_no),
        )
        if predicted is None:
            last_frame, last = history[-1]
            if (
                int(frame_no) - int(last_frame)
                > self.max_misses + 1
            ):
                return None
            predicted = last.centroid_xy
            velocity = (0.0, 0.0)

        vx, vy = velocity
        return TrajectoryPrediction(
            predicted_xy=(
                float(predicted[0]),
                float(predicted[1]),
            ),
            velocity_xy=(float(vx), float(vy)),
            speed_px_per_frame=float(math.hypot(vx, vy)),
            last_component=history[-1][1],
            history_depth=len(history),
        )

    def select(
        self,
        frame_no: int,
        components: list[BallComponent],
        motion_frames: Iterable[
            tuple[int, tuple[BallComponent, ...]]
        ],
    ) -> PredictedTopKSelection:
        frame_no = int(frame_no)
        self._reset_bootstrap_scale_telemetry()
        bootstrap_straightness: float | None = None
        bootstrapped = False

        if not self.locked:
            bootstrap_straightness, _ = self._bootstrap(
                frame_no,
                components,
                motion_frames,
            )
            bootstrapped = self.locked

        prediction = self.prediction(frame_no)
        if prediction is None:
            return PredictedTopKSelection(
                state="UNLOCKED",
                predicted_xy=None,
                gate_px=None,
                considered=len(components),
                selected=(),
                rejected_by_distance=len(components),
                history_depth=0,
                bootstrap_straightness=bootstrap_straightness,
                bootstrap_scale_rejections=(
                    self._bootstrap_scale_rejections
                ),
                bootstrap_scale_observed_px=(
                    self._bootstrap_scale_observed_px
                ),
                bootstrap_scale_expected_px=(
                    self._bootstrap_scale_expected_px
                ),
                bootstrap_scale_ratio=(
                    self._bootstrap_scale_ratio
                ),
            )

        last = prediction.last_component
        gate = self._gate_px(last)
        ranked = []
        for component in components:
            if not self._compatible_scale(last, component):
                continue
            distance = float(
                math.dist(
                    prediction.predicted_xy,
                    component.centroid_xy,
                )
            )
            if distance > gate:
                continue
            scale_ratio = float(component.scale_px) / max(
                0.5,
                float(last.scale_px),
            )
            ranked.append(
                (
                    distance,
                    1 if component.recovered else 0,
                    abs(math.log(max(scale_ratio, 1e-6))),
                    component,
                )
            )

        ranked.sort(key=lambda item: item[:3])
        selected = tuple(
            item[3] for item in ranked[: self.top_k]
        )

        if selected:
            chosen = selected[0]
            if (
                not self._history
                or self._history[-1][0] != frame_no
            ):
                self._history.append((frame_no, chosen))
            else:
                self._history[-1] = (frame_no, chosen)
            self._misses = 0
            state = "BOOTSTRAP" if bootstrapped else "LOCKED"
        else:
            self._misses += 1
            state = "PREDICTED"
            if self._misses > self.max_misses:
                self.reset()
                state = "UNLOCKED"

        return PredictedTopKSelection(
            state=state,
            predicted_xy=prediction.predicted_xy,
            gate_px=float(gate),
            considered=len(components),
            selected=selected,
            rejected_by_distance=max(
                0,
                len(components) - len(ranked),
            ),
            history_depth=self.history_depth,
            bootstrap_straightness=bootstrap_straightness,
            bootstrap_scale_rejections=(
                self._bootstrap_scale_rejections
            ),
            bootstrap_scale_observed_px=(
                self._bootstrap_scale_observed_px
            ),
            bootstrap_scale_expected_px=(
                self._bootstrap_scale_expected_px
            ),
            bootstrap_scale_ratio=(
                self._bootstrap_scale_ratio
            ),
        )
