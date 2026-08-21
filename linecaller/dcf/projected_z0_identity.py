from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    CalibrationCoverage,
    ExternalFrameResult,
    ExternalGridCalibration,
    ExternalGridFrameLoop,
    LockedBallColorProfile,
    Z0Candidate,
    rasterized_scale_compatible,
)


@dataclass(frozen=True)
class ProjectedBallSignature:
    """Five floor-contact hypotheses per exterior cell.

    A pickleball is close to rotationally symmetric at this scale, so the
    signature is geometric + appearance evidence rather than a rotated texture
    atlas.  The selected ball still owns hue/fill identity through
    LockedBallColorProfile.
    """

    cell_id: int
    anchor: str
    floor_xy_bu: tuple[float, float]
    image_xy: tuple[float, float]
    expected_diameter_px: float


@dataclass(frozen=True)
class ProjectionMatch:
    signature: ProjectedBallSignature
    score: float
    anchor_distance_bu: float
    scale_ratio: float
    fill_error: float


@dataclass(frozen=True)
class ApproachEvidence:
    accepted: bool
    prior_observations: int
    predicted_xy: tuple[float, float] | None
    prediction_error_px: float | None
    speed_px_per_frame: float
    total_motion_px: float


class ProjectedZ0SignatureBank:
    """Precomputed CENTER/NORTH/SOUTH/EAST/WEST Z=0 ball hypotheses."""

    _ANCHORS = (
        ("CENTER", 0.50, 0.50),
        ("NORTH", 0.50, 0.20),
        ("SOUTH", 0.50, 0.80),
        ("EAST", 0.80, 0.50),
        ("WEST", 0.20, 0.50),
    )

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        ball_profile: LockedBallColorProfile,
        *,
        max_anchor_distance_bu: float = 0.95,
        min_scale_ratio: float = 0.50,
        max_scale_ratio: float = 1.75,
    ) -> None:
        self.calibration = calibration
        self.ball_profile = ball_profile
        self.max_anchor_distance_bu = float(max_anchor_distance_bu)
        self.min_scale_ratio = float(min_scale_ratio)
        self.max_scale_ratio = float(max_scale_ratio)

        if self.max_anchor_distance_bu <= 0.0:
            raise ValueError("max_anchor_distance_bu must be > 0")
        if not (0.0 < self.min_scale_ratio <= self.max_scale_ratio):
            raise ValueError("invalid projected signature scale range")

        coverage = CalibrationCoverage.parse(calibration.coverage)
        court_x = float(calibration.config.court_x_bu)
        court_y = float(calibration.config.court_y_bu(coverage))
        court_quad = np.asarray(
            (
                (0.0, court_y),
                (court_x, court_y),
                (court_x, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        )
        image_quad = np.asarray(calibration.image_points, dtype=np.float32)
        self._top_to_image = cv2.getPerspectiveTransform(court_quad, image_quad)

        by_cell: dict[int, tuple[ProjectedBallSignature, ...]] = {}
        for cell in calibration.cells:
            x0, y0, x1, y1 = (float(v) for v in cell.top_view_rect_bu)
            signatures: list[ProjectedBallSignature] = []
            for anchor, fx, fy in self._ANCHORS:
                x = x0 + fx * (x1 - x0)
                y = y0 + fy * (y1 - y0)
                image_xy = self._project(x, y)
                signatures.append(
                    ProjectedBallSignature(
                        cell_id=int(cell.cell_id),
                        anchor=anchor,
                        floor_xy_bu=(x, y),
                        image_xy=image_xy,
                        expected_diameter_px=self._local_ball_diameter_px(x, y),
                    )
                )
            by_cell[int(cell.cell_id)] = tuple(signatures)
        self._by_cell = by_cell

    def _project(self, x_bu: float, y_bu: float) -> tuple[float, float]:
        p = np.asarray([[[float(x_bu), float(y_bu)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self._top_to_image)[0, 0]
        return float(q[0]), float(q[1])

    def _local_ball_diameter_px(self, x_bu: float, y_bu: float) -> float:
        # One Ball Unit is one ball diameter.  Use the local projected lateral
        # width, matching the established ExternalGridCell calibration model.
        left = self._project(float(x_bu) - 0.5, float(y_bu))
        right = self._project(float(x_bu) + 0.5, float(y_bu))
        return max(0.5, float(math.dist(left, right)))

    @property
    def signature_count(self) -> int:
        return sum(len(v) for v in self._by_cell.values())

    def signatures_for_cell(self, cell_id: int) -> tuple[ProjectedBallSignature, ...]:
        return self._by_cell.get(int(cell_id), ())

    @staticmethod
    def _component_fill_ratio(component: BallComponent) -> float:
        _, _, w, h = component.bbox
        return float(component.area) / float(max(1, int(w) * int(h)))

    def best_match(
        self,
        candidate: Z0Candidate,
        component: BallComponent,
    ) -> ProjectionMatch | None:
        signatures = self.signatures_for_cell(candidate.cell_id)
        if not signatures:
            return None

        component_fill = self._component_fill_ratio(component)
        template_fill = float(getattr(self.ball_profile, "template_fill_ratio", 0.65))
        fill_error = abs(component_fill - template_fill) / max(0.20, template_fill)

        best: ProjectionMatch | None = None
        for signature in signatures:
            expected = max(0.5, float(signature.expected_diameter_px))
            distance_px = math.dist(candidate.centroid_xy, signature.image_xy)
            distance_bu = distance_px / expected
            scale_ratio = float(candidate.observed_scale_px) / expected
            if not rasterized_scale_compatible(
                candidate.observed_scale_px,
                expected,
                self.min_scale_ratio,
                self.max_scale_ratio,
            ):
                continue
            if distance_bu > self.max_anchor_distance_bu:
                continue

            # Position + local Z0 scale are dominant.  Template fill is a weak
            # appearance cue because a fast ball can smear into a core+halo.
            score = (
                distance_bu
                + 0.70 * abs(math.log(max(scale_ratio, 1e-6)))
                + 0.12 * min(fill_error, 2.0)
            )
            match = ProjectionMatch(
                signature=signature,
                score=float(score),
                anchor_distance_bu=float(distance_bu),
                scale_ratio=float(scale_ratio),
                fill_error=float(fill_error),
            )
            if best is None or match.score < best.score:
                best = match
        return best


class ShortMotionMemory:
    """Short image-space continuity gate.

    This does not create or traverse an interior court grid.  It only remembers
    ball-like image observations from the previous few frames and asks whether
    the current Z0 hypothesis is the plausible continuation of something that
    was already moving toward it.
    """

    def __init__(
        self,
        *,
        history_frames: int = 3,
        min_prior_observations: int = 2,
        prediction_radius_bu: float = 2.50,
        min_total_motion_bu: float = 0.65,
    ) -> None:
        self.history_frames = int(history_frames)
        self.min_prior_observations = int(min_prior_observations)
        self.prediction_radius_bu = float(prediction_radius_bu)
        self.min_total_motion_bu = float(min_total_motion_bu)
        if self.history_frames < 2:
            raise ValueError("history_frames must be >= 2")
        if not (1 <= self.min_prior_observations <= self.history_frames):
            raise ValueError("min_prior_observations must be in [1, history_frames]")
        if self.prediction_radius_bu <= 0.0:
            raise ValueError("prediction_radius_bu must be > 0")
        if self.min_total_motion_bu <= 0.0:
            raise ValueError("min_total_motion_bu must be > 0")
        self._frames: deque[tuple[int, tuple[BallComponent, ...]]] = deque(
            maxlen=self.history_frames
        )

    @property
    def frame_depth(self) -> int:
        return len(self._frames)

    def remember(self, frame_no: int, components: list[BallComponent]) -> None:
        self._frames.append((int(frame_no), tuple(components)))

    @staticmethod
    def _compatible_scale(a: BallComponent, b: BallComponent) -> bool:
        lo = min(float(a.scale_px), float(b.scale_px))
        hi = max(float(a.scale_px), float(b.scale_px))
        if lo <= 0.0:
            return False
        return (hi / lo) <= 2.60

    def _prior_chain(
        self,
        frame_no: int,
        current: BallComponent,
    ) -> list[tuple[int, BallComponent]]:
        chain_reverse: list[tuple[int, BallComponent]] = []
        target = current

        for prior_frame, components in reversed(self._frames):
            if int(frame_no) - int(prior_frame) > self.history_frames + 1:
                continue
            if not components:
                continue

            candidates: list[tuple[float, BallComponent]] = []
            for component in components:
                if not self._compatible_scale(target, component):
                    continue
                distance = math.dist(target.centroid_xy, component.centroid_xy)
                gate = max(
                    5.0,
                    4.75 * max(float(target.scale_px), float(component.scale_px)),
                )
                if distance <= gate:
                    candidates.append((distance, component))
            if not candidates:
                continue

            _, chosen = min(candidates, key=lambda item: item[0])
            chain_reverse.append((int(prior_frame), chosen))
            target = chosen

        return list(reversed(chain_reverse))

    @staticmethod
    def _linear_prediction(
        prior: list[tuple[int, BallComponent]],
        frame_no: int,
    ) -> tuple[tuple[float, float] | None, float]:
        if len(prior) < 2:
            return None, 0.0

        t = np.asarray([float(f) for f, _ in prior], dtype=np.float64)
        xs = np.asarray([float(c.centroid_xy[0]) for _, c in prior], dtype=np.float64)
        ys = np.asarray([float(c.centroid_xy[1]) for _, c in prior], dtype=np.float64)
        tt = t - t.mean()
        denom = float(np.dot(tt, tt))
        if denom <= 1e-9:
            return None, 0.0

        vx = float(np.dot(tt, xs - xs.mean()) / denom)
        vy = float(np.dot(tt, ys - ys.mean()) / denom)
        dt = float(frame_no) - float(t.mean())
        predicted = (float(xs.mean() + vx * dt), float(ys.mean() + vy * dt))
        return predicted, float(math.hypot(vx, vy))

    def evaluate(
        self,
        frame_no: int,
        current: BallComponent,
        *,
        expected_floor_diameter_px: float,
    ) -> ApproachEvidence:
        prior = self._prior_chain(frame_no, current)
        if len(prior) < self.min_prior_observations:
            return ApproachEvidence(
                accepted=False,
                prior_observations=len(prior),
                predicted_xy=None,
                prediction_error_px=None,
                speed_px_per_frame=0.0,
                total_motion_px=0.0,
            )

        oldest_frame, oldest = prior[0]
        frame_span = max(1, int(frame_no) - int(oldest_frame))
        total_motion = float(math.dist(oldest.centroid_xy, current.centroid_xy))
        min_motion_px = max(
            1.5,
            self.min_total_motion_bu * max(0.5, float(expected_floor_diameter_px)),
        )
        if total_motion < min_motion_px:
            return ApproachEvidence(
                accepted=False,
                prior_observations=len(prior),
                predicted_xy=None,
                prediction_error_px=None,
                speed_px_per_frame=total_motion / frame_span,
                total_motion_px=total_motion,
            )

        predicted, fitted_speed = self._linear_prediction(prior, frame_no)
        if predicted is None:
            return ApproachEvidence(
                accepted=False,
                prior_observations=len(prior),
                predicted_xy=None,
                prediction_error_px=None,
                speed_px_per_frame=total_motion / frame_span,
                total_motion_px=total_motion,
            )

        error = float(math.dist(predicted, current.centroid_xy))
        allowed_error = max(
            4.0,
            self.prediction_radius_bu
            * max(0.5, float(expected_floor_diameter_px)),
        )
        return ApproachEvidence(
            accepted=error <= allowed_error,
            prior_observations=len(prior),
            predicted_xy=predicted,
            prediction_error_px=error,
            speed_px_per_frame=max(fitted_speed, total_motion / frame_span),
            total_motion_px=total_motion,
        )


class ProjectedIdentityExternalGridFrameLoop(ExternalGridFrameLoop):
    """Strict live loop used by the Wizard/API.

    Historical ExternalGridFrameLoop semantics stay available for lower-level
    compatibility.  This subclass adds:
      1. five projected Z0 ball signatures per exterior cell;
      2. short motion continuity/prediction before a Z0 candidate is accepted.

    Prediction can focus evidence, but it can never produce OUT by itself.
    """

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        background_rgb,
        ball_profile: LockedBallColorProfile,
        *,
        use_projected_signatures: bool = True,
        require_approach_memory: bool = True,
        approach_history_frames: int = 3,
        approach_min_prior_observations: int = 2,
        projection_anchor_distance_bu: float = 0.95,
        approach_prediction_radius_bu: float = 2.50,
        approach_min_total_motion_bu: float = 0.65,
        **kwargs,
    ) -> None:
        super().__init__(
            calibration,
            background_rgb,
            ball_profile,
            **kwargs,
        )
        self.use_projected_signatures = bool(use_projected_signatures)
        self.require_approach_memory = bool(require_approach_memory)
        self._signature_bank = ProjectedZ0SignatureBank(
            calibration,
            ball_profile,
            max_anchor_distance_bu=float(projection_anchor_distance_bu),
        )
        self._motion = ShortMotionMemory(
            history_frames=int(approach_history_frames),
            min_prior_observations=int(approach_min_prior_observations),
            prediction_radius_bu=float(approach_prediction_radius_bu),
            min_total_motion_bu=float(approach_min_total_motion_bu),
        )
        self._reset_strict_telemetry()

    def _reset_strict_telemetry(self) -> None:
        self._last_projected_matches = 0
        self._last_projected_rejections = 0
        self._last_approach_rejections = 0
        self._last_motion_observations = 0
        self._last_predicted_z0_cell: int | None = None

    @staticmethod
    def _component_for_candidate(
        candidate: Z0Candidate,
        components: list[BallComponent],
    ) -> BallComponent | None:
        if not components:
            return None
        return min(
            components,
            key=lambda component: math.dist(
                candidate.centroid_xy,
                component.centroid_xy,
            ),
        )

    def _find_z0_candidates(
        self,
        frame_no: int,
        components: list[BallComponent],
    ) -> tuple[list[Z0Candidate], int]:
        self._reset_strict_telemetry()
        raw_candidates, boundary_guard_rejections = super()._find_z0_candidates(
            frame_no,
            components,
        )

        accepted: list[Z0Candidate] = []
        for candidate in raw_candidates:
            component = self._component_for_candidate(candidate, components)
            if component is None:
                continue

            projection = None
            if self.use_projected_signatures:
                projection = self._signature_bank.best_match(candidate, component)
                if projection is None:
                    self._last_projected_rejections += 1
                    continue
                self._last_projected_matches += 1

            expected = (
                projection.signature.expected_diameter_px
                if projection is not None
                else candidate.expected_floor_scale_px
            )
            evidence = self._motion.evaluate(
                frame_no,
                component,
                expected_floor_diameter_px=max(0.5, float(expected)),
            )
            self._last_motion_observations = max(
                self._last_motion_observations,
                int(evidence.prior_observations),
            )

            if self.require_approach_memory and not evidence.accepted:
                self._last_approach_rejections += 1
                continue

            if evidence.accepted:
                self._last_predicted_z0_cell = int(candidate.cell_id)
            accepted.append(candidate)

        # Remember ALL ball-like footprints after evaluating this frame so the
        # current observation cannot satisfy its own history requirement.
        self._motion.remember(frame_no, components)
        return accepted, boundary_guard_rejections

    def process_frame(self, frame_no: int, current_rgb) -> ExternalFrameResult:
        result = super().process_frame(frame_no, current_rgb)

        trace = list(result.stage_trace)
        if "Z0_CANDIDATE" in trace:
            index = trace.index("Z0_CANDIDATE")
            trace[index:index] = [
                "PROJECTED_Z0_SIGNATURE",
                "APPROACH_MEMORY",
            ]

        return replace(
            result,
            stage_trace=tuple(trace),
            projected_signature_matches=int(self._last_projected_matches),
            projected_signature_rejections=int(self._last_projected_rejections),
            approach_rejections=int(self._last_approach_rejections),
            motion_history_observations=int(self._last_motion_observations),
            predicted_z0_cell=self._last_predicted_z0_cell,
        )
