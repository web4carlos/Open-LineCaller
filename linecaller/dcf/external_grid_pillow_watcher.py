from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageChops


COURT_WIDTH_BU = 83.0
COURT_LENGTH_BU = 182.6
HALF_COURT_LENGTH_BU = COURT_LENGTH_BU / 2.0


@dataclass(frozen=True)
class ExternalCell:
    cell_id: int
    gx: int
    gy: int
    polygon: tuple[tuple[float, float], ...]
    visible_area_px: int


@dataclass(frozen=True)
class BallColorProfile:
    hue: float
    saturation: float
    value: float
    hue_tolerance: float
    saturation_tolerance: float
    value_tolerance: float

    @staticmethod
    def _hue_delta(h: np.ndarray, center: float) -> np.ndarray:
        d = np.abs(h.astype(np.float32) - float(center))
        return np.minimum(d, 255.0 - d)

    @classmethod
    def from_template(cls, template_path: str | Path) -> "BallColorProfile":
        im = Image.open(template_path).convert("HSV")
        a = np.asarray(im, dtype=np.uint8).reshape(-1, 3)
        if len(a) == 0:
            raise ValueError("Empty ball template.")

        # The template is intentionally tight. Learn the dominant colorful foreground;
        # no hardcoded color name is used.
        s = a[:, 1].astype(np.float32)
        v = a[:, 2].astype(np.float32)

        sat_floor = max(25.0, float(np.percentile(s, 45)))
        val_floor = max(40.0, float(np.percentile(v, 35)))
        keep = (s >= sat_floor) & (v >= val_floor)
        if int(keep.sum()) < max(6, len(a) // 12):
            keep = np.ones(len(a), dtype=bool)

        h = a[keep, 0].astype(np.float32)
        s = a[keep, 1].astype(np.float32)
        v = a[keep, 2].astype(np.float32)

        # Circular hue center from vector mean in PIL's 0..255 hue space.
        theta = h * (2.0 * math.pi / 255.0)
        c = float(np.mean(np.cos(theta)))
        q = float(np.mean(np.sin(theta)))
        hue = (math.atan2(q, c) % (2.0 * math.pi)) * (255.0 / (2.0 * math.pi))

        hdev = cls._hue_delta(h, hue)
        htol = max(10.0, min(38.0, float(np.percentile(hdev, 90)) + 8.0))
        stol = max(45.0, min(125.0, float(np.percentile(np.abs(s - np.median(s)), 90)) + 35.0))
        vtol = max(45.0, min(125.0, float(np.percentile(np.abs(v - np.median(v)), 90)) + 35.0))

        return cls(
            hue=float(hue),
            saturation=float(np.median(s)),
            value=float(np.median(v)),
            hue_tolerance=htol,
            saturation_tolerance=stol,
            value_tolerance=vtol,
        )

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"), dtype=np.uint8)
        h = hsv[..., 0].astype(np.float32)
        s = hsv[..., 1].astype(np.float32)
        v = hsv[..., 2].astype(np.float32)
        return (
            (self._hue_delta(h, self.hue) <= self.hue_tolerance)
            & (np.abs(s - self.saturation) <= self.saturation_tolerance)
            & (np.abs(v - self.value) <= self.value_tolerance)
        )


@dataclass(frozen=True)
class CellCandidate:
    cell_id: int
    gx: int
    gy: int
    ball_pixels: int
    changed_pixels: int
    visible_area_px: int
    score: float
    polygon: tuple[tuple[float, float], ...]


class ExternalGridCalibration:
    """
    CALIBRATION-owned external floor grid.

    Runtime never invents/rebuilds the grid. Runtime only loads this saved
    calibration and asks its already-projected cells about the current frame.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        mode: str,
        image_points: np.ndarray,
        margin_bu: int,
        cells: list[ExternalCell],
        reference_image: str,
    ):
        self.image_size = image_size
        self.mode = mode
        self.image_points = image_points.astype(np.float32)
        self.margin_bu = int(margin_bu)
        self.cells = cells
        self.reference_image = reference_image

    @staticmethod
    def _court_length(mode: str) -> float:
        m = str(mode).upper()
        if "HALF" in m:
            return HALF_COURT_LENGTH_BU
        return COURT_LENGTH_BU

    @staticmethod
    def _homography(image_points: np.ndarray, mode: str) -> np.ndarray:
        y_max = ExternalGridCalibration._court_length(mode)
        world = np.array(
            [
                [0.0, y_max],                 # near-left
                [COURT_WIDTH_BU, y_max],      # near-right
                [COURT_WIDTH_BU, 0.0],        # far-right
                [0.0, 0.0],                   # far-left
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(world, image_points.astype(np.float32))

    @classmethod
    def build(
        cls,
        *,
        image_size: tuple[int, int],
        mode: str,
        image_points: Iterable[Iterable[float]],
        margin_bu: int,
        reference_image: str,
    ) -> "ExternalGridCalibration":
        points = np.asarray(list(image_points), dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("image_points must be 4x2 in near-left, near-right, far-right, far-left order.")

        width, height = int(image_size[0]), int(image_size[1])
        y_max = cls._court_length(mode)
        x_cells = int(round(COURT_WIDTH_BU))
        y_cells = int(round(y_max))
        H = cls._homography(points, mode)

        def project(poly_world: np.ndarray) -> np.ndarray:
            p = cv2.perspectiveTransform(poly_world.reshape(-1, 1, 2).astype(np.float32), H)
            return p.reshape(-1, 2)

        cells: list[ExternalCell] = []
        cid = 0
        margin = int(margin_bu)

        for gy in range(-margin, y_cells + margin):
            for gx in range(-margin, x_cells + margin):
                inside_court = (0 <= gx < x_cells) and (0 <= gy < y_cells)
                if inside_court:
                    continue

                world_poly = np.array(
                    [
                        [float(gx), float(gy)],
                        [float(gx + 1), float(gy)],
                        [float(gx + 1), float(gy + 1)],
                        [float(gx), float(gy + 1)],
                    ],
                    dtype=np.float32,
                )
                poly = project(world_poly)

                # Keep only cells whose projected polygon touches the image.
                xs, ys = poly[:, 0], poly[:, 1]
                if xs.max() < 0 or ys.max() < 0 or xs.min() >= width or ys.min() >= height:
                    continue

                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 1)
                area = int(mask.sum())
                if area <= 0:
                    continue

                cells.append(
                    ExternalCell(
                        cell_id=cid,
                        gx=gx,
                        gy=gy,
                        polygon=tuple((float(x), float(y)) for x, y in poly),
                        visible_area_px=area,
                    )
                )
                cid += 1

        return cls(
            image_size=(width, height),
            mode=str(mode),
            image_points=points,
            margin_bu=margin,
            cells=cells,
            reference_image=str(reference_image),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        payload = {
            "version": 1,
            "owner": "COURT_CALIBRATION",
            "mode": self.mode,
            "image_size": list(self.image_size),
            "image_points_order": ["near-left", "near-right", "far-right", "far-left"],
            "image_points": self.image_points.tolist(),
            "margin_bu": self.margin_bu,
            "reference_image": self.reference_image,
            "cells": [
                {
                    "cell_id": c.cell_id,
                    "gx": c.gx,
                    "gy": c.gy,
                    "polygon": [list(q) for q in c.polygon],
                    "visible_area_px": c.visible_area_px,
                }
                for c in self.cells
            ],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExternalGridCalibration":
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        base = p.parent
        ref = Path(d["reference_image"])
        if not ref.is_absolute():
            ref = base / ref
        cells = [
            ExternalCell(
                cell_id=int(c["cell_id"]),
                gx=int(c["gx"]),
                gy=int(c["gy"]),
                polygon=tuple((float(q[0]), float(q[1])) for q in c["polygon"]),
                visible_area_px=int(c["visible_area_px"]),
            )
            for c in d["cells"]
        ]
        return cls(
            image_size=(int(d["image_size"][0]), int(d["image_size"][1])),
            mode=str(d["mode"]),
            image_points=np.asarray(d["image_points"], dtype=np.float32),
            margin_bu=int(d["margin_bu"]),
            cells=cells,
            reference_image=str(ref),
        )


class ExternalGridPillowWatcher:
    """
    GAME runtime order is deliberately fixed:

        FRAME
          -> PILLOW(current vs CALIBRATED reference)
          -> EXTERNAL GRID
          -> candidate cell(s)

    This RC does NOT track the ball, infer path/velocity/XYZ, or claim bounce.
    It only validates that the fixed calibration-owned grid can raise a local
    cell candidate when the calibrated background changes in a ball-compatible way.
    """

    def __init__(
        self,
        calibration: ExternalGridCalibration,
        ball_profile: BallColorProfile,
        *,
        diff_threshold: int = 18,
        min_ball_pixels: int = 2,
        min_score: float = 0.025,
    ):
        self.calibration = calibration
        self.ball_profile = ball_profile
        self.diff_threshold = int(diff_threshold)
        self.min_ball_pixels = int(min_ball_pixels)
        self.min_score = float(min_score)

        ref = Image.open(calibration.reference_image).convert("RGB")
        if ref.size != calibration.image_size:
            raise ValueError(
                f"Reference image size {ref.size} != calibration image size {calibration.image_size}"
            )
        self.reference = ref

        width, height = calibration.image_size
        self._cell_id_map = np.zeros((height, width), dtype=np.int32)
        self._cells_by_id = {}
        for cell in calibration.cells:
            poly = np.round(np.asarray(cell.polygon, dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(self._cell_id_map, poly, int(cell.cell_id) + 1)
            self._cells_by_id[int(cell.cell_id) + 1] = cell

    def detect(self, frame_rgb: np.ndarray) -> list[CellCandidate]:
        current = Image.fromarray(frame_rgb.astype(np.uint8), "RGB")
        if current.size != self.reference.size:
            raise ValueError("Frame size does not match calibration.")

        # PILLOW comparison is always against the CALIBRATED reference.
        diff = ImageChops.difference(current, self.reference)
        diff_np = np.asarray(diff, dtype=np.uint8)
        changed = diff_np.max(axis=2) >= self.diff_threshold

        ball_like = self.ball_profile.mask(frame_rgb)
        evidence = changed & ball_like & (self._cell_id_map > 0)

        ids = self._cell_id_map[evidence]
        changed_ids = self._cell_id_map[changed & (self._cell_id_map > 0)]
        if ids.size == 0:
            return []

        ball_counts = np.bincount(ids)
        changed_counts = np.bincount(changed_ids) if changed_ids.size else np.zeros(1, dtype=np.int64)

        out: list[CellCandidate] = []
        for encoded_id in np.nonzero(ball_counts)[0]:
            if encoded_id == 0:
                continue
            cell = self._cells_by_id.get(int(encoded_id))
            if cell is None:
                continue
            bp = int(ball_counts[encoded_id])
            cp = int(changed_counts[encoded_id]) if encoded_id < len(changed_counts) else 0
            if bp < self.min_ball_pixels:
                continue
            score = bp / max(1.0, math.sqrt(float(cell.visible_area_px)))
            if score < self.min_score:
                continue
            out.append(
                CellCandidate(
                    cell_id=cell.cell_id,
                    gx=cell.gx,
                    gy=cell.gy,
                    ball_pixels=bp,
                    changed_pixels=cp,
                    visible_area_px=cell.visible_area_px,
                    score=float(score),
                    polygon=cell.polygon,
                )
            )

        out.sort(key=lambda c: (c.score, c.ball_pixels), reverse=True)
        return out
