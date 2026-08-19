from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np

from .models import CourtFrame, Point3D, TimedPoint3D, Velocity3D


class CalibrationMode(str, Enum):
    """Court geometry visible to one calibrated camera/view."""

    FULL_COURT = "FULL_COURT"
    HALF_NEAR = "HALF_NEAR"
    HALF_FAR = "HALF_FAR"


@dataclass(frozen=True)
class ProjectionMeasurement:
    """One trusted 2-D ball projection at a real time.

    center_uv_px is the RC7.2 residual center. apparent_scale_px is image-space
    evidence for the projected ball size. Neither value is itself XYZ.
    """

    center_uv_px: tuple[float, float]
    apparent_scale_px: float
    t_s: float

    def __post_init__(self) -> None:
        u, v = self.center_uv_px
        if not all(math.isfinite(q) for q in (u, v, self.apparent_scale_px, self.t_s)):
            raise ValueError("projection measurement values must be finite")
        if self.apparent_scale_px <= 0.0:
            raise ValueError("apparent_scale_px must be > 0")


@dataclass(frozen=True)
class CourtProjectionCalibration:
    """Projection geometry for FULL_COURT or one HALF_COURT view.

    image_points order is the existing LineCaller order for the visible court
    region: near-left, near-right, far-right, far-left.

    The model intentionally avoids claiming full camera intrinsics. It uses the
    calibrated court quadrilateral plus the locked one-ball spatial scale as a
    lightweight runtime projection model. A later camera model may replace the
    internals without changing the P(x,y,z,t) contract.
    """

    image_points: np.ndarray
    mode: CalibrationMode = CalibrationMode.FULL_COURT
    width_bu: float = CourtFrame().width_bu
    full_length_bu: float = CourtFrame().length_bu

    def __post_init__(self) -> None:
        pts = np.asarray(self.image_points, dtype=np.float64)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            raise ValueError("image_points must be finite 4x2")
        if self.width_bu <= 0.0 or self.full_length_bu <= 0.0:
            raise ValueError("court dimensions must be > 0")
        object.__setattr__(self, "image_points", pts)

    @property
    def y_min_bu(self) -> float:
        if self.mode == CalibrationMode.HALF_NEAR:
            return self.full_length_bu / 2.0
        return 0.0

    @property
    def y_max_bu(self) -> float:
        if self.mode == CalibrationMode.HALF_FAR:
            return self.full_length_bu / 2.0
        return self.full_length_bu

    @property
    def visible_length_bu(self) -> float:
        return self.y_max_bu - self.y_min_bu

    @classmethod
    def from_json_data(cls, data: dict) -> "CourtProjectionCalibration":
        raw_mode = str(data.get("mode", data.get("calibration_mode", "FULL_COURT"))).upper()
        aliases = {
            "FULL": CalibrationMode.FULL_COURT,
            "FULL_COURT": CalibrationMode.FULL_COURT,
            "HALF_NEAR": CalibrationMode.HALF_NEAR,
            "NEAR_HALF": CalibrationMode.HALF_NEAR,
            "HALF_FAR": CalibrationMode.HALF_FAR,
            "FAR_HALF": CalibrationMode.HALF_FAR,
        }
        if raw_mode not in aliases:
            raise ValueError(f"unsupported calibration mode: {raw_mode}")
        return cls(np.asarray(data["image_points"], dtype=np.float64), aliases[raw_mode])

    def _fraction_for_y(self, y_bu: float) -> float:
        if self.visible_length_bu <= 0.0:
            raise ValueError("invalid visible court length")
        return (float(y_bu) - self.y_min_bu) / self.visible_length_bu

    def _edges_at_y(self, y_bu: float) -> tuple[np.ndarray, np.ndarray]:
        # s=0 is the FAR edge of this view; s=1 is the NEAR edge.
        s = self._fraction_for_y(y_bu)
        nl, nr, fr, fl = self.image_points
        left = fl * (1.0 - s) + nl * s
        right = fr * (1.0 - s) + nr * s
        return left, right

    def expected_one_bu_scale_px(self, y_bu: float) -> float:
        left, right = self._edges_at_y(y_bu)
        return float(np.linalg.norm(right - left) / self.width_bu)

    def floor_point(self, x_bu: float, y_bu: float) -> np.ndarray:
        left, right = self._edges_at_y(y_bu)
        u = float(x_bu) / self.width_bu
        return left * (1.0 - u) + right * u

    def project_xyz(self, point: Point3D) -> tuple[float, float]:
        floor = self.floor_point(point.x, point.y)
        scale = self.expected_one_bu_scale_px(point.y)
        # Existing runtime DCF convention: +Z projects upward in the image.
        return float(floor[0]), float(floor[1] - point.z * scale)

    def y_from_normalized_scale(self, normalized_scale_px: float) -> float:
        if not math.isfinite(normalized_scale_px) or normalized_scale_px <= 0.0:
            raise ValueError("normalized scale must be finite and > 0")
        # Robust numerical inverse. Court quadrilateral perspective normally
        # makes scale monotonic, but nearest-search stays safe if it is not.
        ys = np.linspace(self.y_min_bu, self.y_max_bu, 2049)
        scales = np.array([self.expected_one_bu_scale_px(float(y)) for y in ys])
        idx = int(np.argmin(np.abs(scales - normalized_scale_px)))
        return float(ys[idx])

    def unproject_with_scale(
        self,
        center_uv_px: tuple[float, float],
        apparent_scale_px: float,
        *,
        scale_gain: float,
    ) -> Point3D:
        if not math.isfinite(scale_gain) or scale_gain <= 0.0:
            raise ValueError("scale_gain must be finite and > 0")
        u_px, v_px = map(float, center_uv_px)
        normalized = apparent_scale_px / scale_gain
        y = self.y_from_normalized_scale(normalized)
        left, right = self._edges_at_y(y)
        edge = right - left
        denom = float(np.dot(edge, edge))
        if denom <= 1e-12:
            raise ValueError("degenerate court width projection")
        # Screen-up Z is orthogonal to the horizontal court-width evidence in
        # the current runtime model, so project onto the width vector for X.
        q = np.array([u_px, v_px], dtype=np.float64)
        frac_x = float(np.dot(q - left, edge) / denom)
        x = frac_x * self.width_bu
        floor = left * (1.0 - frac_x) + right * frac_x
        scale = self.expected_one_bu_scale_px(y)
        if scale <= 1e-12:
            raise ValueError("degenerate local projection scale")
        z = max(0.0, float((floor[1] - v_px) / scale))
        return Point3D(x, y, z)


@dataclass(frozen=True)
class XYZResolution:
    measurement: ProjectionMeasurement
    raw_projection_xyz: Point3D
    resolved: TimedPoint3D
    predicted_from_continuity: Point3D | None
    scale_gain: float
    center_reprojection_error_px: float
    scale_residual_px: float
    calibration_mode: CalibrationMode


class ContinuousXYZResolver:
    """Resolve one continuous P(x,y,z,t) from RC7.2 projection evidence.

    Key rule: the residual center/scale are measurements; they do not define a
    flight path. The resolver combines those measurements with the previous
    same-ball state and time to choose one continuous XYZ solution.

    scale_gain is a runtime nuisance factor that maps RC7.2's irregular
    residual-size evidence to the calibrated one-BU projection scale. It is
    primed once from a trusted same-ball anchor and then kept fixed for the
    flight epoch. This avoids treating a partial/non-circular residual as a
    literal physical diameter.
    """

    def __init__(
        self,
        calibration: CourtProjectionCalibration,
        *,
        measurement_weight: float = 0.72,
        max_speed_bu_s: float = 400.0,
        max_z_bu: float = 96.0,
    ) -> None:
        if not 0.0 < measurement_weight <= 1.0:
            raise ValueError("measurement_weight must be in (0,1]")
        if max_speed_bu_s <= 0.0 or max_z_bu <= 0.0:
            raise ValueError("speed and Z limits must be > 0")
        self.calibration = calibration
        self.measurement_weight = float(measurement_weight)
        self.max_speed_bu_s = float(max_speed_bu_s)
        self.max_z_bu = float(max_z_bu)
        self._scale_gain: float | None = None
        self._history: list[TimedPoint3D] = []

    @property
    def scale_gain(self) -> float | None:
        return self._scale_gain

    @property
    def history(self) -> tuple[TimedPoint3D, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._scale_gain = None
        self._history.clear()

    def prime(self, anchor: TimedPoint3D, measurement: ProjectionMeasurement) -> XYZResolution:
        if abs(anchor.t_s - measurement.t_s) > 1e-6:
            raise ValueError("anchor and measurement timestamps must match")
        expected = self.calibration.expected_one_bu_scale_px(anchor.y)
        gain = measurement.apparent_scale_px / expected
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("could not establish runtime scale gain")
        self._scale_gain = float(gain)
        self._history = [anchor]
        raw = self.calibration.unproject_with_scale(
            measurement.center_uv_px,
            measurement.apparent_scale_px,
            scale_gain=gain,
        )
        return self._make_resolution(measurement, raw, anchor, None)

    def _predict(self, t_s: float) -> Point3D | None:
        if not self._history:
            return None
        last = self._history[-1]
        if t_s <= last.t_s:
            raise ValueError("measurement time must increase")
        if len(self._history) < 2:
            return last.point
        prev = self._history[-2]
        v = Velocity3D.between(prev, last)
        dt = t_s - last.t_s
        return Point3D(
            last.x + v.vx * dt,
            last.y + v.vy * dt,
            max(0.0, last.z + v.vz * dt),
        )

    @staticmethod
    def _clamp_delta(value: float, center: float, radius: float) -> float:
        return min(center + radius, max(center - radius, value))

    def resolve(self, measurement: ProjectionMeasurement) -> XYZResolution:
        if self._scale_gain is None or not self._history:
            raise RuntimeError("resolver must be primed with one trusted anchor")
        pred = self._predict(measurement.t_s)
        raw = self.calibration.unproject_with_scale(
            measurement.center_uv_px,
            measurement.apparent_scale_px,
            scale_gain=self._scale_gain,
        )
        if pred is None:
            point = raw
        else:
            dt = measurement.t_s - self._history[-1].t_s
            radius = max(0.75, self.max_speed_bu_s * dt)
            # First enforce physical same-ball continuity, then blend the
            # projection measurement with the local motion prediction.
            rx = self._clamp_delta(raw.x, pred.x, radius)
            ry = self._clamp_delta(raw.y, pred.y, radius)
            rz = self._clamp_delta(raw.z, pred.z, radius)
            w = self.measurement_weight
            point = Point3D(
                w * rx + (1.0 - w) * pred.x,
                min(self.calibration.y_max_bu, max(self.calibration.y_min_bu, w * ry + (1.0 - w) * pred.y)),
                min(self.max_z_bu, max(0.0, w * rz + (1.0 - w) * pred.z)),
            )
        timed = TimedPoint3D(point, measurement.t_s)
        self._history.append(timed)
        # Resolver itself needs only two prior states. Keep bounded history.
        if len(self._history) > 3:
            self._history = self._history[-3:]
        return self._make_resolution(measurement, raw, timed, pred)

    def _make_resolution(
        self,
        measurement: ProjectionMeasurement,
        raw: Point3D,
        resolved: TimedPoint3D,
        pred: Point3D | None,
    ) -> XYZResolution:
        assert self._scale_gain is not None
        pu, pv = self.calibration.project_xyz(resolved.point)
        mu, mv = measurement.center_uv_px
        reproj = math.hypot(pu - mu, pv - mv)
        expected_scale = (
            self.calibration.expected_one_bu_scale_px(resolved.y) * self._scale_gain
        )
        return XYZResolution(
            measurement=measurement,
            raw_projection_xyz=raw,
            resolved=resolved,
            predicted_from_continuity=pred,
            scale_gain=self._scale_gain,
            center_reprojection_error_px=float(reproj),
            scale_residual_px=float(measurement.apparent_scale_px - expected_scale),
            calibration_mode=self.calibration.mode,
        )
