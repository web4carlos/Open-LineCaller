from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibrator,
    ExternalGridConfig,
)


def load_court_calibration(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data["image_points"]
    raw_mode = data.get("mode", data.get("calibration_mode", "FULL_COURT"))
    try:
        coverage = CalibrationCoverage.parse(raw_mode)
    except ValueError:
        coverage = CalibrationCoverage.FULL_COURT
    return data, pts, coverage


def read_frame(video: str, frame_no: int):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read background frame {frame_no}")
    return frame


def draw_top_view(cal, out: Path):
    cfg = cal.config
    court_y = cfg.court_y_bu(CalibrationCoverage.parse(cal.coverage))
    margin = cfg.margin_bu
    scale = 4.0
    w = int(round((cfg.court_x_bu + 2 * margin) * scale))
    h = int(round((court_y + 2 * margin) * scale))
    img = Image.new("RGB", (max(1, w), max(1, h)), "black")
    d = ImageDraw.Draw(img)

    def xy(x, y):
        return ((x + margin) * scale, (y + margin) * scale)

    for c in cal.cells:
        x0, y0, x1, y1 = c.top_view_rect_bu
        d.rectangle([xy(x0, y0), xy(x1, y1)], outline=(90, 90, 90), width=1)
    d.rectangle([xy(0, 0), xy(cfg.court_x_bu, court_y)], outline=(255, 255, 255), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def draw_perspective(background_bgr, cal, out: Path):
    img = background_bgr.copy()
    for c in cal.cells:
        pts = np.asarray([[int(round(x)), int(round(y))] for x, y in c.polygon_image], dtype=np.int32)
        cv2.polylines(img, [pts], True, (100, 100, 100), 1, cv2.LINE_AA)
    court = np.asarray([[int(round(x)), int(round(y))] for x, y in cal.image_points], dtype=np.int32)
    cv2.polylines(img, [court], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "CALIBRATION: 2D EXTERNAL GRID -> CAMERA PERSPECTIVE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
    ux, uy = cal.image_up_unit
    p0 = (48, 58)
    p1 = (int(round(p0[0] + ux * 28)), int(round(p0[1] + uy * 28)))
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=.25)
    cv2.putText(img, "UP", (p1[0] + 4, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--court-calibration", required=True)
    ap.add_argument("--background-frame", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--margin-bu", type=float, default=10.0)
    ap.add_argument("--up-x", type=float, default=0.0)
    ap.add_argument("--up-y", type=float, default=-1.0)
    a = ap.parse_args()

    _, pts, coverage = load_court_calibration(a.court_calibration)
    bg = read_frame(a.video, a.background_frame)
    h, w = bg.shape[:2]
    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bg_path = outdir / "EXTERNAL_GRID_BACKGROUND.png"
    cv2.imwrite(str(bg_path), bg)

    cfg = ExternalGridConfig(margin_bu=float(a.margin_bu))
    builder = ExternalGridCalibrator(
        pts,
        (w, h),
        coverage=coverage,
        config=cfg,
        image_up_unit=(a.up_x, a.up_y),
    )
    cal = builder.build(background_image=bg_path.name)
    json_path = outdir / "EXTERNAL_GRID_CALIBRATION.json"
    cal.save(json_path)
    draw_top_view(cal, outdir / "EXTERNAL_GRID_TOP_VIEW.png")
    draw_perspective(bg, cal, outdir / "EXTERNAL_GRID_PERSPECTIVE.png")

    print("============================================================")
    print(" RC8.2 CALIBRATION - EXTERNAL GRID + UP")
    print("============================================================")
    print(f"COURT MODE....................... {coverage.value}")
    print("GRID OWNER....................... COURT CALIBRATION")
    print("GRID SOURCE...................... 2D TOP VIEW")
    print("GRID PERSPECTIVE................. PROJECTED DURING CALIBRATION")
    print("WATCHED AREA..................... EXTERNAL ONLY")
    print(f"LOGICAL CELL..................... {cfg.cell_bu:.2f} BALL UNIT")
    print(f"EXTERNAL MARGIN.................. {cfg.margin_bu:.2f} BALL UNITS")
    print(f"VISIBLE EXTERNAL CELLS........... {len(cal.cells)}")
    print(f"CALIBRATED IMAGE UP.............. ({cal.image_up_unit[0]:.3f},{cal.image_up_unit[1]:.3f})")
    print("RUNTIME GRID CREATION............ NO")
    print(f"calibration={json_path}")
    print(f"background={bg_path}")
    print(f"top_view={outdir / 'EXTERNAL_GRID_TOP_VIEW.png'}")
    print(f"perspective={outdir / 'EXTERNAL_GRID_PERSPECTIVE.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
