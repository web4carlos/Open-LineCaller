from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.one_ball_color_field import OneBallColorDCF, DCFCell


def read_frame(video, frame_no):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_no}")
    return frame


def build_bridge_cells(frame_shape, quad, cell_px):
    """
    Compatibility bridge only.

    The scanner is now strict CP-0035.9.2. These cells are still produced by
    the legacy image grid so that we can A/B test immediately on the existing
    calibration. The next geometry step will replace this builder with real
    projected DCF cells.
    """
    h, w = frame_shape[:2]
    far_y = float(min(quad[:, 1]))
    near_y = float(max(quad[:, 1]))
    cells = []

    for ix, x1 in enumerate(range(0, w, cell_px)):
        for iy, y1 in enumerate(range(0, h, cell_px)):
            x2 = min(w, x1 + cell_px)
            y2 = min(h, y1 + cell_px)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            inside = cv2.pointPolygonTest(
                quad.reshape((-1, 1, 2)),
                (float(cx), float(cy)),
                False,
            ) >= 0

            depth = max(
                0.0,
                min(1.0, (cy - far_y) / max(1.0, near_y - far_y)),
            )
            scale = 0.45 + 1.65 * depth

            cells.append(
                DCFCell(ix, iy, x1, y1, x2, y2, inside, scale)
            )

    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cell-px", type=int, default=48)
    p.add_argument("--min-score", type=float, default=.58)
    p.add_argument("--min-color-score", type=float, default=.50)
    a = p.parse_args()

    frame = read_frame(a.video, a.frame)
    template = cv2.imread(a.template)
    if template is None:
        raise RuntimeError("Could not read template")

    data = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    quad = np.asarray(data["image_points"], dtype=np.float32)
    cells = build_bridge_cells(frame.shape, quad, a.cell_px)

    scanner = OneBallColorDCF(
        a.min_score,
        min_color_score=a.min_color_score,
    )
    hit = scanner.search(frame, template, cells)

    vis = frame.copy()
    if hit.found:
        cv2.circle(
            vis,
            (round(hit.x), round(hit.y)),
            12,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            vis,
            (round(hit.cell.center_x), round(hit.cell.center_y)),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"frame_{a.frame}_true_dcf_cell_scanner.jpg"
    cv2.imwrite(str(fn), vis)

    print("CP-0035.9.2 TRUE DCF CELL SCANNER")
    print("NOTE: scanner=true-cell; builder=legacy-grid bridge")
    print(f"cells_scanned={len(cells)}")
    print(f"ball_locked={scanner.ball_locked}")
    print(f"ball_generation={scanner.ball_generation}")
    print(f"ball_found={hit.found}")
    print(f"score={hit.score:.6f}")
    print(f"template_score={hit.template_score:.6f}")
    print(f"color_score={hit.color_score:.6f}")
    print(f"center_score={hit.center_score:.6f}")
    print(f"offset_px={hit.offset_px}")
    print(f"x={hit.x}")
    print(f"y={hit.y}")
    print(f"cell={None if hit.cell is None else (hit.cell.ix, hit.cell.iy)}")
    print(f"inside_internal_mesh={None if hit.cell is None else hit.cell.inside}")
    print(f"image={fn}")


if __name__ == "__main__":
    main()
