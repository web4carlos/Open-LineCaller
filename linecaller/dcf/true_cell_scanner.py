from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class DCFCell:
    """
    A DCF cell is a physical hypothesis projected into image space.

    CP-0035.9.2 keeps the legacy rectangular bounds for compatibility.
    The scanner does NOT search the whole rectangle.  It searches only a
    very small neighbourhood around the projected cell centre.
    """
    ix: int
    iy: int
    x1: int
    y1: int
    x2: int
    y2: int
    inside: bool
    expected_scale: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class BallColorProfile:
    hue: float
    saturation: float
    value: float
    hue_tolerance: float
    saturation_tolerance: float
    value_tolerance: float
    sample_count: int

    @staticmethod
    def _circular_hue_center(hues: np.ndarray) -> float:
        angles = hues.astype(np.float64) * (2.0 * math.pi / 180.0)
        s = float(np.sin(angles).mean())
        c = float(np.cos(angles).mean())
        angle = math.atan2(s, c)
        if angle < 0:
            angle += 2.0 * math.pi
        return angle * 180.0 / (2.0 * math.pi)

    @staticmethod
    def _hue_distance(hues: np.ndarray, center: float) -> np.ndarray:
        diff = np.abs(hues.astype(np.float64) - center)
        return np.minimum(diff, 180.0 - diff)

    @classmethod
    def from_reference(cls, reference_bgr: np.ndarray) -> "BallColorProfile":
        if reference_bgr is None or reference_bgr.size == 0:
            raise ValueError("Ball reference is empty")

        hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Prefer the central ellipse.  A ball template often contains some
        # background around the ball, and that background must not become
        # part of the locked identity.
        yy, xx = np.ogrid[:h, :w]
        rx = max(1.0, w * 0.42)
        ry = max(1.0, h * 0.42)
        ellipse = (((xx - (w - 1) / 2.0) / rx) ** 2 +
                   ((yy - (h - 1) / 2.0) / ry) ** 2) <= 1.0

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        vivid = ellipse & (sat >= 25) & (val >= 35)
        mask = vivid if int(vivid.sum()) >= 8 else ellipse

        pixels = hsv[mask]
        if pixels.size == 0:
            pixels = hsv.reshape(-1, 3)

        hues = pixels[:, 0].astype(np.float64)
        sats = pixels[:, 1].astype(np.float64)
        vals = pixels[:, 2].astype(np.float64)

        hue_center = cls._circular_hue_center(hues)
        sat_center = float(np.median(sats))
        val_center = float(np.median(vals))

        hue_dev = cls._hue_distance(hues, hue_center)
        sat_dev = np.abs(sats - sat_center)
        val_dev = np.abs(vals - val_center)

        # Floors let the profile survive normal exposure/white-balance changes.
        hue_tol = max(8.0, float(np.percentile(hue_dev, 90)) + 4.0)
        sat_tol = max(35.0, float(np.percentile(sat_dev, 90)) + 18.0)
        val_tol = max(40.0, float(np.percentile(val_dev, 90)) + 20.0)

        return cls(
            hue=float(hue_center),
            saturation=sat_center,
            value=val_center,
            hue_tolerance=hue_tol,
            saturation_tolerance=sat_tol,
            value_tolerance=val_tol,
            sample_count=int(len(pixels)),
        )

    def similarity(self, patch_bgr: np.ndarray) -> float:
        other = BallColorProfile.from_reference(patch_bgr)

        hue_diff = min(abs(other.hue - self.hue), 180.0 - abs(other.hue - self.hue))
        sat_diff = abs(other.saturation - self.saturation)
        val_diff = abs(other.value - self.value)

        hue_score = max(0.0, 1.0 - hue_diff / max(12.0, self.hue_tolerance * 2.0))
        sat_score = max(0.0, 1.0 - sat_diff / max(55.0, self.saturation_tolerance * 2.0))
        val_score = max(0.0, 1.0 - val_diff / max(65.0, self.value_tolerance * 2.0))

        return float(0.60 * hue_score + 0.25 * sat_score + 0.15 * val_score)


@dataclass(frozen=True)
class ColorMatch:
    found: bool
    score: float
    x: float | None
    y: float | None
    cell: DCFCell | None
    template_score: float = 0.0
    color_score: float = 0.0
    center_score: float = 0.0
    offset_px: float | None = None
    ball_generation: int = 0


class TrueDCFCellScanner:
    """
    CP-0035.9.2 -- True DCF Cell Scanner

    Rule:
        ONE ACTIVE BALL -> ONE DCF -> ONE CELL HYPOTHESIS

    Important:
      * The DCF cell predicts WHERE the ball centre should be.
      * expected_scale predicts HOW LARGE the ball should appear.
      * The scanner does not search freely through a full 48x48 cell.
      * The first supplied ball reference becomes LOCKED.
      * A different ball can only become active through replace_ball().

    This class intentionally preserves the CP-0035.9.1 public search signature:
        search(frame, ball_reference, cells)
    """

    def __init__(
        self,
        min_score: float = 0.58,
        *,
        min_color_score: float = 0.50,
        center_tolerance_ratio: float = 0.30,
        min_center_tolerance_px: int = 3,
        max_center_tolerance_px: int = 8,
    ):
        self.min_score = float(min_score)
        self.min_color_score = float(min_color_score)
        self.center_tolerance_ratio = float(center_tolerance_ratio)
        self.min_center_tolerance_px = int(min_center_tolerance_px)
        self.max_center_tolerance_px = int(max_center_tolerance_px)

        self._ball_reference: np.ndarray | None = None
        self._color_profile: BallColorProfile | None = None
        self._ball_generation = 0

    @property
    def ball_locked(self) -> bool:
        return self._ball_reference is not None and self._color_profile is not None

    @property
    def ball_generation(self) -> int:
        return self._ball_generation

    @property
    def color_profile(self) -> BallColorProfile | None:
        return self._color_profile

    def lock_ball(self, ball_reference: np.ndarray) -> BallColorProfile:
        if ball_reference is None or ball_reference.size == 0:
            raise ValueError("Ball reference is empty")
        self._ball_reference = ball_reference.copy()
        self._color_profile = BallColorProfile.from_reference(self._ball_reference)
        if self._ball_generation == 0:
            self._ball_generation = 1
        return self._color_profile

    def replace_ball(self, ball_reference: np.ndarray) -> BallColorProfile:
        """
        Explicit Ball Replacement Event.

        Court calibration and DCF remain untouched; only the active visual
        identity changes.
        """
        if ball_reference is None or ball_reference.size == 0:
            raise ValueError("Ball reference is empty")
        self._ball_generation = max(1, self._ball_generation + 1)
        self._ball_reference = ball_reference.copy()
        self._color_profile = BallColorProfile.from_reference(self._ball_reference)
        return self._color_profile

    def clear_ball(self) -> None:
        self._ball_reference = None
        self._color_profile = None

    @staticmethod
    def _resize_template(template: np.ndarray, scale: float) -> np.ndarray:
        h, w = template.shape[:2]
        tw = max(3, round(w * float(scale)))
        th = max(3, round(h * float(scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(template, (tw, th), interpolation=interp)

    @staticmethod
    def _joint_color_corr(crop: np.ndarray, tpl: np.ndarray) -> tuple[float, tuple[int, int]]:
        """
        One shared response map for B/G/R.

        CP-0035.9.1 took the maximum independently in each channel and then
        returned only the location from the last channel.  Here all channels
        must agree on ONE image location.
        """
        responses = []
        for ch in range(3):
            response = cv2.matchTemplate(
                crop[:, :, ch],
                tpl[:, :, ch],
                cv2.TM_CCOEFF_NORMED,
            )
            responses.append(response)

        joint = np.mean(np.stack(responses, axis=0), axis=0)
        _, mx, _, loc = cv2.minMaxLoc(joint)
        return float(max(0.0, mx)), (int(loc[0]), int(loc[1]))

    def _tolerance_px(self, tw: int, th: int) -> int:
        diameter = min(tw, th)
        tol = round(diameter * self.center_tolerance_ratio)
        return max(
            self.min_center_tolerance_px,
            min(self.max_center_tolerance_px, int(tol)),
        )

    @staticmethod
    def _bounded_roi(
        frame_shape,
        center_x: float,
        center_y: float,
        tpl_w: int,
        tpl_h: int,
        tolerance: int,
    ) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        half_w = tpl_w / 2.0 + tolerance
        half_h = tpl_h / 2.0 + tolerance

        x1 = max(0, int(math.floor(center_x - half_w)))
        y1 = max(0, int(math.floor(center_y - half_h)))
        x2 = min(w, int(math.ceil(center_x + half_w)))
        y2 = min(h, int(math.ceil(center_y + half_h)))
        return x1, y1, x2, y2

    def search(self, frame, ball_reference, cells):
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        # Backward-compatible first-call behaviour: the first reference becomes
        # the active locked ball. Subsequent search() calls cannot silently
        # change identity; replace_ball() is required.
        if not self.ball_locked:
            self.lock_ball(ball_reference)

        assert self._ball_reference is not None
        assert self._color_profile is not None

        best = None

        for cell in cells:
            tpl = self._resize_template(self._ball_reference, cell.expected_scale)
            th, tw = tpl.shape[:2]
            tolerance = self._tolerance_px(tw, th)

            # TRUE CELL SEMANTICS:
            # Search only around the projected centre of THIS hypothesis.
            rx1, ry1, rx2, ry2 = self._bounded_roi(
                frame.shape,
                cell.center_x,
                cell.center_y,
                tw,
                th,
                tolerance,
            )

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0 or tw > crop.shape[1] or th > crop.shape[0]:
                continue

            corr, loc = self._joint_color_corr(crop, tpl)
            x0, y0 = loc

            patch = crop[y0:y0 + th, x0:x0 + tw]
            if patch.shape[:2] != tpl.shape[:2]:
                continue

            cx = rx1 + x0 + tw / 2.0
            cy = ry1 + y0 + th / 2.0

            dx = cx - cell.center_x
            dy = cy - cell.center_y
            offset = math.hypot(dx, dy)

            # Hard physical gate: a match outside the small projected-centre
            # tolerance is NOT this DCF cell even if template correlation is high.
            if offset > tolerance + 0.75:
                continue

            center_score = max(0.0, 1.0 - offset / max(1.0, float(tolerance)))
            color_score = self._color_profile.similarity(patch)

            # Locked color is a gate, not merely a small bonus.
            if color_score < self.min_color_score:
                continue

            final_score = (
                0.55 * corr +
                0.30 * color_score +
                0.15 * center_score
            )

            candidate = (
                float(final_score),
                float(cx),
                float(cy),
                cell,
                float(corr),
                float(color_score),
                float(center_score),
                float(offset),
            )

            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            return ColorMatch(
                False, 0.0, None, None, None,
                ball_generation=self._ball_generation,
            )

        (
            score, x, y, cell,
            template_score, color_score, center_score, offset,
        ) = best

        if score < self.min_score:
            return ColorMatch(
                False,
                score,
                None,
                None,
                None,
                template_score,
                color_score,
                center_score,
                offset,
                self._ball_generation,
            )

        return ColorMatch(
            True,
            score,
            x,
            y,
            cell,
            template_score,
            color_score,
            center_score,
            offset,
            self._ball_generation,
        )
