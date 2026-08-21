
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.external_grid_frame_loop import (
    BallComponent,
    ExternalFrameResult,
    ExternalGridFrameLoop,
    Z0Candidate,
    rasterized_scale_compatible,
)
from linecaller.dcf.projected_z0_identity import (
    ApproachEvidence,
    ProjectedIdentityExternalGridFrameLoop,
    ProjectionMatch,
)


@dataclass(frozen=True)
class RecoveryTrackPrediction:
    predicted_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    speed_px_per_frame: float
    prior_observations: int
    last_component: BallComponent


class ContactRecoveryProjectedIdentityExternalGridFrameLoop(
    ProjectedIdentityExternalGridFrameLoop
):
    """CP-0036.2.4.2 strict live loop.

    Contact recovery is deliberately narrow:
    - a locked-color trajectory must already exist;
    - it must be moving physically DOWN in calibrated image space;
    - its next position must land near a projected Z0 signature owned by the
      active outside-only runtime calibration;
    - only then may a bright/desaturated changed component temporarily stand
      in for the selected ball color for a small number of frames.

    Recovery itself never creates OUT. The recovered component must still pass
    external-cell geometry, projected Z0 identity, approach memory, line-touch
    safety and later UP confirmation.
    """

    def __init__(
        self,
        *args,
        contact_appearance_recovery: bool = True,
        contact_recovery_max_frames: int = 2,
        contact_recovery_prediction_radius_bu: float = 1.60,
        contact_recovery_component_radius_bu: float = 1.35,
        contact_recovery_min_down_bu_per_frame: float = 0.05,
        contact_recovery_min_value_ratio: float = 0.55,
        contact_recovery_max_candidates: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contact_appearance_recovery = bool(contact_appearance_recovery)
        self.contact_recovery_max_frames = int(contact_recovery_max_frames)
        self.contact_recovery_prediction_radius_bu = float(
            contact_recovery_prediction_radius_bu
        )
        self.contact_recovery_component_radius_bu = float(
            contact_recovery_component_radius_bu
        )
        self.contact_recovery_min_down_bu_per_frame = float(
            contact_recovery_min_down_bu_per_frame
        )
        self.contact_recovery_min_value_ratio = float(
            contact_recovery_min_value_ratio
        )
        self.contact_recovery_max_candidates = int(
            contact_recovery_max_candidates
        )
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
            raise ValueError("contact_recovery_max_candidates must be >= 1")

        signatures = [
            signature
            for cell_signatures in self._signature_bank._by_cell.values()
            for signature in cell_signatures
        ]
        self._recovery_signatures = tuple(signatures)
        if signatures:
            self._recovery_signature_xy = np.asarray(
                [s.image_xy for s in signatures],
                dtype=np.float64,
            )
            self._recovery_signature_diameter = np.asarray(
                [max(0.5, float(s.expected_diameter_px)) for s in signatures],
                dtype=np.float64,
            )
        else:
            self._recovery_signature_xy = np.empty((0, 2), dtype=np.float64)
            self._recovery_signature_diameter = np.empty((0,), dtype=np.float64)

        self._processing_frame_no: int | None = None
        self._recovery_changed_mask: np.ndarray | None = None
        self._recovery_hsv: np.ndarray | None = None
        self._reset_contact_recovery_telemetry()
        self._reset_approach_diagnostics()

    def _reset_contact_recovery_telemetry(self) -> None:
        self._last_contact_recoveries = 0
        self._last_contact_recovery_frame: int | None = None
        self._last_contact_recovery_cell: int | None = None

    def _reset_approach_diagnostics(self) -> None:
        self._last_approach_reason: str | None = None
        self._last_approach_total_motion_px = 0.0
        self._last_approach_min_motion_px = 0.0
        self._last_approach_prediction_error_px: float | None = None
        self._last_approach_allowed_error_px: float | None = None
        self._last_candidate_cell: int | None = None
        self._last_projected_anchor: str | None = None

    def _components(
        self,
        current: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, list[BallComponent]]:
        changed, ball_changed, components = super()._components(current)
        if self.contact_appearance_recovery:
            self._recovery_changed_mask = changed
            self._recovery_hsv = np.asarray(
                current.convert("HSV"),
                dtype=np.uint8,
            )
        return changed, ball_changed, components

    def _merge_motion_components(
        self,
        components: list[BallComponent],
    ) -> list[BallComponent]:
        merged = super()._merge_motion_components(components)
        if (
            not self.contact_appearance_recovery
            or self._processing_frame_no is None
            or self._recovery_changed_mask is None
            or self._recovery_hsv is None
        ):
            return merged
        recovered = self._recover_contact_components(
            self._processing_frame_no,
            merged,
        )
        return [*merged, *recovered]

    def _motion_predictions(
        self,
        frame_no: int,
    ) -> list[RecoveryTrackPrediction]:
        frames = [
            (int(f), tuple(components))
            for f, components in self._motion._frames
            if int(f) < int(frame_no)
            and int(frame_no) - int(f) <= self._motion.history_frames + 1
        ]
        if len(frames) < 2:
            return []

        latest_frame, latest_components = frames[-1]
        if int(frame_no) - latest_frame > 2:
            return []

        predictions: list[RecoveryTrackPrediction] = []
        for last in latest_components:
            if (
                last.recovered
                and int(last.recovery_age) >= self.contact_recovery_max_frames
            ):
                continue

            chain_reverse: list[tuple[int, BallComponent]] = [
                (latest_frame, last)
            ]
            target = last
            for prior_frame, prior_components in reversed(frames[:-1]):
                if latest_frame - int(prior_frame) > self._motion.history_frames:
                    continue

                candidates: list[tuple[float, BallComponent]] = []
                for component in prior_components:
                    if not self._motion._compatible_scale(target, component):
                        continue
                    distance = math.dist(
                        target.centroid_xy,
                        component.centroid_xy,
                    )
                    gate = max(
                        5.0,
                        4.75
                        * max(
                            float(target.scale_px),
                            float(component.scale_px),
                        ),
                    )
                    if distance <= gate:
                        candidates.append((distance, component))

                if not candidates:
                    continue
                _, chosen = min(candidates, key=lambda item: item[0])
                chain_reverse.append((int(prior_frame), chosen))
                target = chosen
                if len(chain_reverse) >= self._motion.history_frames:
                    break

            chain = list(reversed(chain_reverse))
            if len(chain) < self._motion.min_prior_observations:
                continue

            predicted, speed = self._motion._linear_prediction(
                chain,
                int(frame_no),
            )
            if predicted is None:
                continue

            dt = max(1, int(frame_no) - latest_frame)
            vx = (
                float(predicted[0]) - float(last.centroid_xy[0])
            ) / float(dt)
            vy = (
                float(predicted[1]) - float(last.centroid_xy[1])
            ) / float(dt)
            predictions.append(
                RecoveryTrackPrediction(
                    predicted_xy=predicted,
                    velocity_xy=(vx, vy),
                    speed_px_per_frame=max(
                        float(speed),
                        float(math.hypot(vx, vy)),
                    ),
                    prior_observations=len(chain),
                    last_component=last,
                )
            )
        return predictions

    def _nearest_signature(
        self,
        point_xy: tuple[float, float],
    ):
        if not self._recovery_signatures:
            return None, math.inf

        p = np.asarray(point_xy, dtype=np.float64)
        delta = self._recovery_signature_xy - p
        distances_px = np.sqrt(np.sum(delta * delta, axis=1))
        distances_bu = (
            distances_px
            / np.maximum(0.5, self._recovery_signature_diameter)
        )
        index = int(np.argmin(distances_bu))
        return (
            self._recovery_signatures[index],
            float(distances_bu[index]),
        )

    def _recover_changed_component(
        self,
        prediction: RecoveryTrackPrediction,
        signature,
    ) -> BallComponent | None:
        changed = self._recovery_changed_mask
        hsv = self._recovery_hsv
        if changed is None or hsv is None:
            return None

        expected = max(0.5, float(signature.expected_diameter_px))
        radius = max(
            4.0,
            self.contact_recovery_component_radius_bu * expected,
        )
        px, py = prediction.predicted_xy
        height, width = changed.shape[:2]
        x0 = max(0, int(math.floor(px - radius)))
        y0 = max(0, int(math.floor(py - radius)))
        x1 = min(width, int(math.ceil(px + radius)) + 1)
        y1 = min(height, int(math.ceil(py + radius)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        roi = changed[y0:y1, x0:x1].astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            roi,
            8,
        )
        hsv_roi = hsv[y0:y1, x0:x1]
        candidates: list[tuple[float, BallComponent]] = []

        min_value = max(
            65.0,
            float(self.ball_profile.value_min)
            * self.contact_recovery_min_value_ratio,
        )
        relaxed_sat = max(
            90.0,
            float(self.ball_profile.saturation_min) * 1.35,
        )
        expanded_hue = min(
            48.0,
            max(12.0, float(self.ball_profile.hue_tolerance) * 3.0),
        )

        for label in range(1, n):
            rx, ry, rw, rh, area = (
                int(v) for v in stats[label]
            )
            if area < max(2, self.min_color_pixels) or area > 520:
                continue
            if rw < 2 or rh < 2:
                continue
            aspect = max(rw / rh, rh / rw)
            if aspect > 5.5:
                continue

            cx_local, cy_local = (
                float(v) for v in centroids[label]
            )
            cx = float(x0) + cx_local
            cy = float(y0) + cy_local
            distance_px = math.dist(
                (cx, cy),
                prediction.predicted_xy,
            )
            if distance_px > radius:
                continue

            scale = float(min(rw, rh))
            scale_ratio = scale / expected
            if not rasterized_scale_compatible(
                scale,
                expected,
                self.min_floor_scale_ratio,
                self.max_floor_scale_ratio,
            ):
                continue

            mask = labels == label
            sat_values = hsv_roi[..., 1][mask].astype(np.float32)
            val_values = hsv_roi[..., 2][mask].astype(np.float32)
            hue_values = hsv_roi[..., 0][mask].astype(np.float32)
            if not val_values.size:
                continue

            median_sat = float(np.median(sat_values))
            median_val = float(np.median(val_values))
            if median_val < min_value:
                continue

            desaturated_bright = (
                median_sat <= relaxed_sat
                and median_val
                >= max(100.0, float(self.ball_profile.value_min) * 0.70)
            )

            if hue_values.size:
                hue_distance = float(
                    np.median(
                        self.ball_profile._hue_distance(
                            hue_values,
                            self.ball_profile.hue_center,
                        )
                    )
                )
            else:
                hue_distance = 999.0
            relaxed_color = hue_distance <= expanded_hue
            if not (desaturated_bright or relaxed_color):
                continue

            fill = float(area) / float(max(1, rw * rh))
            template_fill = float(
                getattr(self.ball_profile, "template_fill_ratio", 0.65)
            )
            fill_error = abs(fill - template_fill) / max(
                0.20,
                template_fill,
            )
            score = (
                distance_px / expected
                + 0.60 * abs(math.log(max(scale_ratio, 1e-6)))
                + 0.12 * min(fill_error, 2.0)
                + (0.08 if desaturated_bright else 0.0)
            )
            recovery_age = (
                int(prediction.last_component.recovery_age) + 1
                if prediction.last_component.recovered
                else 1
            )
            candidates.append(
                (
                    float(score),
                    BallComponent(
                        bbox=(x0 + rx, y0 + ry, rw, rh),
                        area=int(area),
                        centroid_xy=(cx, cy),
                        scale_px=scale,
                        merged_from=1,
                        recovered=True,
                        recovery_age=recovery_age,
                    ),
                )
            )

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _recover_contact_components(
        self,
        frame_no: int,
        existing_components: list[BallComponent],
    ) -> list[BallComponent]:
        recovered: list[BallComponent] = []
        ux, uy = self.calibration.image_up_unit

        for prediction in self._motion_predictions(frame_no):
            signature, predicted_distance_bu = self._nearest_signature(
                prediction.predicted_xy
            )
            if signature is None:
                continue
            if (
                predicted_distance_bu
                > self.contact_recovery_prediction_radius_bu
            ):
                continue

            expected = max(0.5, float(signature.expected_diameter_px))
            last = prediction.last_component
            last_scale_ratio = float(last.scale_px) / expected
            if not (0.35 <= last_scale_ratio <= 2.60):
                continue

            vx, vy = prediction.velocity_xy
            down_px_per_frame = -(vx * ux + vy * uy)
            min_down_px = (
                self.contact_recovery_min_down_bu_per_frame * expected
            )
            if down_px_per_frame < min_down_px:
                continue

            last_distance_bu = (
                math.dist(last.centroid_xy, signature.image_xy)
                / expected
            )
            if predicted_distance_bu > last_distance_bu + 0.20:
                continue

            duplicate_radius = max(3.0, 0.80 * expected)
            if any(
                math.dist(
                    component.centroid_xy,
                    prediction.predicted_xy,
                )
                <= duplicate_radius
                for component in [*existing_components, *recovered]
            ):
                continue

            component = self._recover_changed_component(
                prediction,
                signature,
            )
            if component is None:
                continue

            recovered.append(component)
            self._last_contact_recoveries += 1
            self._last_contact_recovery_frame = int(frame_no)
            self._last_contact_recovery_cell = int(signature.cell_id)
            if len(recovered) >= self.contact_recovery_max_candidates:
                break

        return recovered

    def _diagnostic_reason(
        self,
        evidence: ApproachEvidence,
        *,
        expected_floor_diameter_px: float,
    ) -> tuple[str, float, float]:
        expected = max(0.5, float(expected_floor_diameter_px))
        min_motion_px = max(
            1.5,
            self._motion.min_total_motion_bu * expected,
        )
        allowed_error_px = max(
            4.0,
            self._motion.prediction_radius_bu * expected,
        )
        if (
            int(evidence.prior_observations)
            < self._motion.min_prior_observations
        ):
            reason = "INSUFFICIENT_HISTORY"
        elif float(evidence.total_motion_px) < min_motion_px:
            reason = "INSUFFICIENT_MOTION"
        elif evidence.predicted_xy is None:
            reason = "NO_LINEAR_PREDICTION"
        elif evidence.accepted:
            reason = "ACCEPTED"
        else:
            reason = "PREDICTION_ERROR"
        return reason, float(min_motion_px), float(allowed_error_px)

    def _record_approach_diagnostics(
        self,
        candidate: Z0Candidate,
        projection: ProjectionMatch | None,
        evidence: ApproachEvidence,
        *,
        expected_floor_diameter_px: float,
    ) -> None:
        reason, min_motion, allowed_error = self._diagnostic_reason(
            evidence,
            expected_floor_diameter_px=expected_floor_diameter_px,
        )
        should_replace = (
            self._last_candidate_cell is None
            or evidence.accepted
            or self._last_approach_reason != "ACCEPTED"
        )
        if not should_replace:
            return

        self._last_approach_reason = reason
        self._last_approach_total_motion_px = float(
            evidence.total_motion_px
        )
        self._last_approach_min_motion_px = min_motion
        self._last_approach_prediction_error_px = (
            None
            if evidence.prediction_error_px is None
            else float(evidence.prediction_error_px)
        )
        self._last_approach_allowed_error_px = allowed_error
        self._last_candidate_cell = int(candidate.cell_id)
        self._last_projected_anchor = (
            projection.signature.anchor
            if projection is not None
            else None
        )

    def _find_z0_candidates(
        self,
        frame_no: int,
        components: list[BallComponent],
    ) -> tuple[list[Z0Candidate], int]:
        self._reset_strict_telemetry()
        self._reset_approach_diagnostics()

        raw_candidates, boundary_guard_rejections = (
            ExternalGridFrameLoop._find_z0_candidates(
                self,
                frame_no,
                components,
            )
        )

        accepted: list[Z0Candidate] = []
        for candidate in raw_candidates:
            component = self._component_for_candidate(
                candidate,
                components,
            )
            if component is None:
                continue

            projection = None
            if self.use_projected_signatures:
                projection = self._signature_bank.best_match(
                    candidate,
                    component,
                )
                if projection is None:
                    self._last_projected_rejections += 1
                    continue
                self._last_projected_matches += 1

            expected = (
                projection.signature.expected_diameter_px
                if projection is not None
                else candidate.expected_floor_scale_px
            )
            expected = max(0.5, float(expected))
            evidence = self._motion.evaluate(
                frame_no,
                component,
                expected_floor_diameter_px=expected,
            )
            self._last_motion_observations = max(
                self._last_motion_observations,
                int(evidence.prior_observations),
            )
            self._record_approach_diagnostics(
                candidate,
                projection,
                evidence,
                expected_floor_diameter_px=expected,
            )

            if self.require_approach_memory and not evidence.accepted:
                self._last_approach_rejections += 1
                continue

            if evidence.accepted:
                self._last_predicted_z0_cell = int(candidate.cell_id)
            accepted.append(candidate)

        self._motion.remember(frame_no, components)
        return accepted, boundary_guard_rejections

    def process_frame(
        self,
        frame_no: int,
        current_rgb,
    ) -> ExternalFrameResult:
        self._processing_frame_no = int(frame_no)
        self._recovery_changed_mask = None
        self._recovery_hsv = None
        self._reset_contact_recovery_telemetry()
        try:
            result = super().process_frame(frame_no, current_rgb)

            trace = list(result.stage_trace)
            if "BALL_MOTION_FOOTPRINT" in trace:
                index = trace.index("BALL_MOTION_FOOTPRINT") + 1
                if "CONTACT_APPEARANCE_RECOVERY" not in trace:
                    trace.insert(index, "CONTACT_APPEARANCE_RECOVERY")

            return replace(
                result,
                stage_trace=tuple(trace),
                approach_reason=self._last_approach_reason,
                approach_total_motion_px=float(
                    self._last_approach_total_motion_px
                ),
                approach_min_motion_px=float(
                    self._last_approach_min_motion_px
                ),
                approach_prediction_error_px=(
                    self._last_approach_prediction_error_px
                ),
                approach_allowed_error_px=(
                    self._last_approach_allowed_error_px
                ),
                candidate_cell=self._last_candidate_cell,
                projected_anchor=self._last_projected_anchor,
                contact_recoveries=int(self._last_contact_recoveries),
                contact_recovery_frame=(
                    self._last_contact_recovery_frame
                ),
                contact_recovery_cell=(
                    self._last_contact_recovery_cell
                ),
            )
        finally:
            self._processing_frame_no = None
            self._recovery_changed_mask = None
            self._recovery_hsv = None
