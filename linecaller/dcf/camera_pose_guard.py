from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridCell,
)


class PoseGuardState(str, Enum):
    SAFE = "SAFE"
    MICRO_ADJUST = "MICRO_ADJUST"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CameraPoseGuardConfig:
    """
    Conservative thresholds for a camera mounted near the net/post.

    Tiny vibration is tolerated. A measurable but still recoverable movement
    requires a micro-adjustment. A large/uncertain movement blocks officiating.
    """

    anchor_band_px: int = 28
    max_features: int = 180
    quality_level: float = 0.01
    min_feature_distance_px: float = 5.0
    min_tracks: int = 12
    max_lk_error: float = 40.0
    ransac_reprojection_px: float = 2.5
    min_inlier_ratio: float = 0.55

    safe_max_corner_shift_px: float = 2.5
    safe_rmse_px: float = 1.5

    micro_adjust_max_corner_shift_px: float = 12.0
    micro_adjust_max_rmse_px: float = 3.5

    def __post_init__(self) -> None:
        if self.anchor_band_px < 4:
            raise ValueError("anchor_band_px must be >= 4")
        if self.max_features < 12:
            raise ValueError("max_features must be >= 12")
        if self.min_tracks < 4:
            raise ValueError("min_tracks must be >= 4")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level must be in (0,1]")
        if self.min_feature_distance_px <= 0.0:
            raise ValueError("min_feature_distance_px must be > 0")
        if self.max_lk_error <= 0.0:
            raise ValueError("max_lk_error must be > 0")
        if self.ransac_reprojection_px <= 0.0:
            raise ValueError("ransac_reprojection_px must be > 0")
        if not 0.0 < self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in (0,1]")
        if self.safe_max_corner_shift_px < 0.0:
            raise ValueError("safe_max_corner_shift_px must be >= 0")
        if self.micro_adjust_max_corner_shift_px < self.safe_max_corner_shift_px:
            raise ValueError(
                "micro_adjust_max_corner_shift_px must be >= safe_max_corner_shift_px"
            )
        if self.safe_rmse_px < 0.0:
            raise ValueError("safe_rmse_px must be >= 0")
        if self.micro_adjust_max_rmse_px < self.safe_rmse_px:
            raise ValueError("micro_adjust_max_rmse_px must be >= safe_rmse_px")


@dataclass(frozen=True)
class CameraPoseDecision:
    state: PoseGuardState
    reason: str
    tracked_points: int
    inlier_points: int
    inlier_ratio: float
    reprojection_rmse_px: float | None
    median_corner_shift_px: float | None
    max_corner_shift_px: float | None
    homography_ref_to_current: np.ndarray | None

    @property
    def allows_out_call(self) -> bool:
        # MICRO_ADJUST is intentionally false until the adjusted geometry and
        # reference are installed and the guard is evaluated again.
        return self.state is PoseGuardState.SAFE

    @property
    def requires_micro_adjustment(self) -> bool:
        return self.state is PoseGuardState.MICRO_ADJUST

    @property
    def blocked(self) -> bool:
        return self.state is PoseGuardState.BLOCKED


@dataclass(frozen=True)
class PoseAdjustedRuntime:
    calibration: ExternalGridCalibration
    background_rgb: Image.Image
    applied_homography_ref_to_current: np.ndarray


def _as_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    a = np.asarray(image, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("image must be RGB HxWx3")
    return Image.fromarray(a, mode="RGB")


def _transform_points(points, H: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(p, H.astype(np.float64))
    return q.reshape(-1, 2).astype(np.float32)


def _normalized_direction_after_homography(
    H: np.ndarray,
    direction: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    w, h = image_size
    ux, uy = float(direction[0]), float(direction[1])
    n = math.hypot(ux, uy)
    if n <= 1e-9:
        raise ValueError("direction must be non-zero")
    ux, uy = ux / n, uy / n

    center = np.array([[0.5 * w, 0.5 * h]], dtype=np.float32)
    tip = np.array([[0.5 * w + ux * 24.0, 0.5 * h + uy * 24.0]], dtype=np.float32)
    moved = _transform_points(np.vstack([center, tip]), H)
    dx = float(moved[1, 0] - moved[0, 0])
    dy = float(moved[1, 1] - moved[0, 1])
    m = math.hypot(dx, dy)
    if m <= 1e-9:
        return (ux, uy)
    return (dx / m, dy / m)


class CameraPoseGuard:
    """
    Guard the calibration against camera movement.

    Reference features are detected ONLY around the four calibrated exterior
    boundary edges/corners. The court interior is not searched as a DCF/grid.

    Processing:
        calibrated reference boundary anchors
        -> LK optical flow into current frame
        -> RANSAC homography
        -> calibrated-corner displacement
        -> SAFE / MICRO_ADJUST / BLOCKED

    Fail-closed behavior is deliberate: weak anchor evidence is BLOCKED.
    """

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        reference_rgb: Image.Image | np.ndarray,
        *,
        config: CameraPoseGuardConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = config or CameraPoseGuardConfig()
        self.reference = _as_rgb_image(reference_rgb)

        if self.reference.size != tuple(calibration.image_size):
            raise ValueError("reference size does not match calibration image_size")

        self._reference_gray = cv2.cvtColor(
            np.asarray(self.reference, dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        self._anchor_mask = self._build_anchor_mask()
        self._reference_features = self._detect_reference_features()

    def _build_anchor_mask(self) -> np.ndarray:
        width, height = self.calibration.image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.rint(
            np.asarray(self.calibration.image_points, dtype=np.float32)
        ).astype(np.int32)

        thickness = int(self.config.anchor_band_px)
        cv2.polylines(
            mask,
            [pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        radius = max(thickness, 10)
        for x, y in pts:
            cv2.circle(mask, (int(x), int(y)), radius, 255, -1, cv2.LINE_AA)
        return mask

    def _detect_reference_features(self) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            self._reference_gray,
            maxCorners=int(self.config.max_features),
            qualityLevel=float(self.config.quality_level),
            minDistance=float(self.config.min_feature_distance_px),
            mask=self._anchor_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32)

    @property
    def reference_feature_count(self) -> int:
        return int(len(self._reference_features))

    def _blocked(
        self,
        reason: str,
        *,
        tracked: int = 0,
        inliers: int = 0,
        inlier_ratio: float = 0.0,
        rmse: float | None = None,
        median_shift: float | None = None,
        max_shift: float | None = None,
        H: np.ndarray | None = None,
    ) -> CameraPoseDecision:
        return CameraPoseDecision(
            state=PoseGuardState.BLOCKED,
            reason=reason,
            tracked_points=int(tracked),
            inlier_points=int(inliers),
            inlier_ratio=float(inlier_ratio),
            reprojection_rmse_px=rmse,
            median_corner_shift_px=median_shift,
            max_corner_shift_px=max_shift,
            homography_ref_to_current=H,
        )

    def evaluate(
        self,
        current_rgb: Image.Image | np.ndarray,
    ) -> CameraPoseDecision:
        current = _as_rgb_image(current_rgb)
        if current.size != self.reference.size:
            return self._blocked("FRAME_SIZE_MISMATCH")

        if self.reference_feature_count < self.config.min_tracks:
            return self._blocked(
                "INSUFFICIENT_REFERENCE_ANCHORS",
                tracked=self.reference_feature_count,
            )

        current_gray = cv2.cvtColor(
            np.asarray(current, dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._reference_gray,
            current_gray,
            self._reference_features,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )

        if next_points is None or status is None:
            return self._blocked("ANCHOR_TRACKING_FAILED")

        status = status.reshape(-1).astype(bool)
        if errors is not None:
            e = errors.reshape(-1)
            status &= np.isfinite(e) & (e <= float(self.config.max_lk_error))

        ref = self._reference_features.reshape(-1, 2)[status]
        cur = next_points.reshape(-1, 2)[status]
        tracked = int(len(ref))

        if tracked < self.config.min_tracks:
            return self._blocked(
                "INSUFFICIENT_TRACKED_ANCHORS",
                tracked=tracked,
            )

        H, inlier_mask = cv2.findHomography(
            ref,
            cur,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.config.ransac_reprojection_px),
        )
        if H is None or inlier_mask is None or not np.isfinite(H).all():
            return self._blocked(
                "POSE_HOMOGRAPHY_FAILED",
                tracked=tracked,
            )

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        ratio = inliers / max(1, tracked)
        if inliers < 4 or ratio < self.config.min_inlier_ratio:
            return self._blocked(
                "LOW_POSE_INLIER_CONFIDENCE",
                tracked=tracked,
                inliers=inliers,
                inlier_ratio=ratio,
                H=H,
            )

        projected = _transform_points(ref[inlier_mask], H)
        residual = projected - cur[inlier_mask]
        rmse = float(
            math.sqrt(
                float(np.mean(np.sum(residual.astype(np.float64) ** 2, axis=1)))
            )
        )

        corners = np.asarray(
            self.calibration.image_points,
            dtype=np.float32,
        )
        moved_corners = _transform_points(corners, H)
        shifts = np.linalg.norm(moved_corners - corners, axis=1)
        median_shift = float(np.median(shifts))
        max_shift = float(np.max(shifts))

        if (
            max_shift <= self.config.safe_max_corner_shift_px
            and rmse <= self.config.safe_rmse_px
        ):
            state = PoseGuardState.SAFE
            reason = "POSE_STABLE"
        elif (
            max_shift <= self.config.micro_adjust_max_corner_shift_px
            and rmse <= self.config.micro_adjust_max_rmse_px
        ):
            state = PoseGuardState.MICRO_ADJUST
            reason = "POSE_MOVED_WITHIN_RECOVERABLE_RANGE"
        else:
            return self._blocked(
                "POSE_SHIFT_EXCEEDS_SAFE_RECOVERY",
                tracked=tracked,
                inliers=inliers,
                inlier_ratio=ratio,
                rmse=rmse,
                median_shift=median_shift,
                max_shift=max_shift,
                H=H,
            )

        return CameraPoseDecision(
            state=state,
            reason=reason,
            tracked_points=tracked,
            inlier_points=inliers,
            inlier_ratio=float(ratio),
            reprojection_rmse_px=rmse,
            median_corner_shift_px=median_shift,
            max_corner_shift_px=max_shift,
            homography_ref_to_current=H.astype(np.float64),
        )


def apply_micro_adjustment(
    calibration: ExternalGridCalibration,
    background_rgb: Image.Image | np.ndarray,
    decision: CameraPoseDecision,
) -> PoseAdjustedRuntime:
    """
    Apply a recoverable camera pose change to the image-side calibration only.

    World/top-view cell rectangles and OUT region identities are never altered.
    Therefore an exterior cell cannot become an interior cell through pose repair.

    The calibrated background is warped by the SAME homography so Pillow/runtime
    comparisons remain aligned with the moved camera.
    """
    if decision.state is not PoseGuardState.MICRO_ADJUST:
        raise ValueError("apply_micro_adjustment requires MICRO_ADJUST decision")
    H = decision.homography_ref_to_current
    if H is None:
        raise ValueError("MICRO_ADJUST decision has no homography")

    background = _as_rgb_image(background_rgb)
    if background.size != tuple(calibration.image_size):
        raise ValueError("background size does not match calibration")

    width, height = calibration.image_size
    warped = cv2.warpPerspective(
        np.asarray(background, dtype=np.uint8),
        H.astype(np.float64),
        (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    new_image_points = _transform_points(calibration.image_points, H)

    new_cells: list[ExternalGridCell] = []
    for cell in calibration.cells:
        poly = _transform_points(cell.polygon_image, H)
        xs = poly[:, 0]
        ys = poly[:, 1]
        bx0 = max(0, int(math.floor(float(xs.min()))) - 1)
        by0 = max(0, int(math.floor(float(ys.min()))) - 1)
        bx1 = min(width, int(math.ceil(float(xs.max()))) + 2)
        by1 = min(height, int(math.ceil(float(ys.max()))) + 2)

        x0, _, x1, _ = cell.top_view_rect_bu
        step_x = max(1e-6, float(x1) - float(x0))
        top_w = float(np.linalg.norm(poly[1] - poly[0])) / step_x
        bot_w = float(np.linalg.norm(poly[2] - poly[3])) / step_x
        expected = max(0.5, 0.5 * (top_w + bot_w))

        new_cells.append(
            ExternalGridCell(
                cell_id=cell.cell_id,
                ix=cell.ix,
                iy=cell.iy,
                region=cell.region,
                top_view_rect_bu=cell.top_view_rect_bu,
                polygon_image=tuple(
                    (float(x), float(y)) for x, y in poly
                ),
                bbox_image=(bx0, by0, bx1, by1),
                expected_floor_ball_diameter_px=expected,
            )
        )

    new_up = _normalized_direction_after_homography(
        H,
        calibration.image_up_unit,
        calibration.image_size,
    )

    adjusted = ExternalGridCalibration(
        version=calibration.version,
        coverage=calibration.coverage,
        image_size=calibration.image_size,
        image_points=tuple(
            (float(x), float(y)) for x, y in new_image_points
        ),
        config=calibration.config,
        background_image=calibration.background_image,
        image_up_unit=new_up,
        cells=tuple(new_cells),
    )

    return PoseAdjustedRuntime(
        calibration=adjusted,
        background_rgb=Image.fromarray(warped, mode="RGB"),
        applied_homography_ref_to_current=H.astype(np.float64),
    )