from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import math

import numpy as np
from PIL import Image, ImageChops

from linecaller.dcf.external_grid_pillow_watcher import BallColorProfile, ExternalCell, ExternalGridCalibration

ActiveSide = Literal["FAR", "NEAR"]

@dataclass(frozen=True)
class CellRuntimeMeta:
    cell_id: int
    gx: int
    gy: int
    polygon: tuple[tuple[float, float], ...]
    floor_position_bu: tuple[float, float, float]
    expected_scale_px: float
    floor_anchor_px: tuple[float, float]
    up_direction_px: tuple[float, float]

@dataclass(frozen=True)
class Z0Candidate:
    frame: int
    cell: CellRuntimeMeta
    center_px: tuple[float, float]
    ball_pixels: int
    scale_px: float
    identity_score: float

@dataclass(frozen=True)
class Z0Confirmation:
    candidate_frame: int
    confirm_frame: int
    cell: CellRuntimeMeta
    up_center_px: tuple[float, float]
    up_pixels: int


def _poly_array(cell):
    return np.asarray(cell.polygon, dtype=np.float32)


def _edge_len(a, b):
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _expected_one_bu_scale(poly):
    xscale = 0.5 * (_edge_len(poly[0], poly[1]) + _edge_len(poly[3], poly[2]))
    yscale = 0.5 * (_edge_len(poly[0], poly[3]) + _edge_len(poly[1], poly[2]))
    return float(math.sqrt(max(1.0, xscale) * max(1.0, yscale)))


def _center(poly):
    c = poly.mean(axis=0)
    return float(c[0]), float(c[1])


def select_side_cells(calibration, side: ActiveSide, *, depth_bu: int = 4):
    s = str(side).upper()
    if s not in {"FAR", "NEAR"}:
        raise ValueError("side must be FAR or NEAR")
    depth = int(depth_bu)
    if depth <= 0:
        raise ValueError("depth_bu must be > 0")

    x_cells = 83
    mode = str(calibration.mode).upper()
    y_cells = 91 if "HALF" in mode else 183
    split = y_cells // 2
    out = []

    for c in calibration.cells:
        gx, gy = int(c.gx), int(c.gy)
        if "HALF" in mode:
            left = (-depth <= gx < 0) and (0 <= gy < y_cells)
            right = (x_cells <= gx < x_cells + depth) and (0 <= gy < y_cells)
            baseline = (((-depth <= gy < 0) if s == "FAR" else (y_cells <= gy < y_cells + depth)) and (0 <= gx < x_cells))
            keep = left or right or baseline
        elif s == "FAR":
            keep = (
                ((-depth <= gy < 0) and (0 <= gx < x_cells))
                or ((-depth <= gx < 0) and (0 <= gy < split))
                or ((x_cells <= gx < x_cells + depth) and (0 <= gy < split))
            )
        else:
            keep = (
                ((y_cells <= gy < y_cells + depth) and (0 <= gx < x_cells))
                or ((-depth <= gx < 0) and (split <= gy < y_cells))
                or ((x_cells <= gx < x_cells + depth) and (split <= gy < y_cells))
            )
        if keep:
            out.append(c)

    if not out:
        raise RuntimeError(f"No visible external cells for side={s}, depth={depth}")
    out.sort(key=lambda c: (int(c.gy), int(c.gx), int(c.cell_id)))
    return out


def build_runtime_meta(cells, *, up_direction_px=(0.0, -1.0)):
    ux, uy = map(float, up_direction_px)
    n = math.hypot(ux, uy)
    if n <= 1e-6:
        raise ValueError("up_direction_px cannot be zero")
    ux, uy = ux / n, uy / n
    out = []
    for c in cells:
        poly = _poly_array(c)
        scale = _expected_one_bu_scale(poly)
        cx, cy = _center(poly)
        out.append(
            CellRuntimeMeta(
                int(c.cell_id), int(c.gx), int(c.gy),
                tuple((float(x), float(y)) for x, y in poly),
                (float(c.gx) + 0.5, float(c.gy) + 0.5, 0.0),
                scale, (cx, cy), (ux, uy)
            )
        )
    return out


class ExternalGridZ0Game:
    def __init__(self, calibration, ball_profile, *, active_side: ActiveSide, depth_bu=4,
                 diff_threshold=18, min_ball_pixels=2, size_ratio_min=0.28, size_ratio_max=2.60,
                 up_frames=3, up_min_bu=0.55, up_max_bu=2.80, up_direction_px=(0.0, -1.0)):
        self.calibration = calibration
        self.ball_profile = ball_profile
        self.active_side = str(active_side).upper()
        self.diff_threshold = int(diff_threshold)
        self.min_ball_pixels = int(min_ball_pixels)
        self.size_ratio_min = float(size_ratio_min)
        self.size_ratio_max = float(size_ratio_max)
        self.up_frames = int(up_frames)
        self.up_min_bu = float(up_min_bu)
        self.up_max_bu = float(up_max_bu)

        ref = Image.open(calibration.reference_image).convert("RGB")
        if ref.size != calibration.image_size:
            raise ValueError("Reference size does not match calibration.")
        self.reference = ref
        self.cells = build_runtime_meta(
            select_side_cells(calibration, self.active_side, depth_bu=depth_bu),
            up_direction_px=up_direction_px,
        )

        import cv2
        h, w = calibration.image_size[1], calibration.image_size[0]
        self._shape = (h, w)
        self._masks = {}
        for c in self.cells:
            mask = np.zeros(self._shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(np.asarray(c.polygon)).astype(np.int32), 1)
            self._masks[c.cell_id] = mask.astype(bool)

        self.pending: Optional[Z0Candidate] = None

    @staticmethod
    def _component_metrics(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0, (0.0, 0.0), 0.0
        cx, cy = float(xs.mean()), float(ys.mean())
        xspan = float(xs.max() - xs.min() + 1)
        yspan = float(ys.max() - ys.min() + 1)
        return int(len(xs)), (cx, cy), float(math.sqrt(max(1.0, xspan * yspan)))

    def _frame_evidence(self, frame_rgb):
        current = Image.fromarray(frame_rgb.astype(np.uint8), "RGB")
        if current.size != self.reference.size:
            raise ValueError("Frame size does not match calibration.")
        diff = np.asarray(ImageChops.difference(current, self.reference), dtype=np.uint8)
        changed = diff.max(axis=2) >= self.diff_threshold
        ball_like = self.ball_profile.mask(frame_rgb)
        return changed, ball_like

    def _candidate_from_cell(self, frame_no, cell, changed, ball_like):
        evidence = changed & ball_like & self._masks[cell.cell_id]
        count, center, scale = self._component_metrics(evidence)
        if count < self.min_ball_pixels:
            return None
        ratio = scale / max(1.0, cell.expected_scale_px)
        if not (self.size_ratio_min <= ratio <= self.size_ratio_max):
            return None
        score = float(count) / max(1.0, cell.expected_scale_px)
        return Z0Candidate(int(frame_no), cell, center, count, scale, score)

    def scan_for_one_candidate(self, frame_no, frame_rgb):
        changed, ball_like = self._frame_evidence(frame_rgb)
        best = None
        for cell in self.cells:
            candidate = self._candidate_from_cell(frame_no, cell, changed, ball_like)
            if candidate is not None and (best is None or candidate.identity_score > best.identity_score):
                best = candidate
        self.pending = best
        return best

    def check_above_after_fact(self, frame_no, frame_rgb):
        p = self.pending
        if p is None:
            return None
        age = int(frame_no) - int(p.frame)
        if age <= 0:
            return None
        if age > self.up_frames:
            self.pending = None
            return None

        changed, ball_like = self._frame_evidence(frame_rgb)
        ys, xs = np.nonzero(changed & ball_like)
        if len(xs) == 0:
            return None

        ux, uy = p.cell.up_direction_px
        scale = max(1.0, p.cell.expected_scale_px)
        px, py = p.center_px
        dx = xs.astype(np.float32) - float(px)
        dy = ys.astype(np.float32) - float(py)
        vertical = dx * ux + dy * uy
        lateral = np.abs(dx * (-uy) + dy * ux)
        keep = (
            (vertical >= self.up_min_bu * scale)
            & (vertical <= self.up_max_bu * scale)
            & (lateral <= 1.75 * scale)
        )
        if int(keep.sum()) < self.min_ball_pixels:
            return None

        kx, ky = xs[keep], ys[keep]
        confirmation = Z0Confirmation(
            p.frame, int(frame_no), p.cell,
            (float(kx.mean()), float(ky.mean())), int(len(kx))
        )
        self.pending = None
        return confirmation
