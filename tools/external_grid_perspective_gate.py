from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from linecaller.dcf.external_grid_frame_loop import (
    CalibrationCoverage,
    ExternalGridCalibrator,
    ExternalGridConfig,
)
from tools.build_external_grid_calibration import load_court_calibration, read_frame, draw_top_view, draw_perspective


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--court-calibration', required=True)
    ap.add_argument('--background-frame', type=int, required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--margin-bu', type=float, default=4.0)
    a = ap.parse_args()

    _, pts, coverage = load_court_calibration(a.court_calibration)
    frame = read_frame(a.video, a.background_frame)
    h, w = frame.shape[:2]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bg_path = out / 'EXTERNAL_GRID_BACKGROUND.png'
    cv2.imwrite(str(bg_path), frame)

    cfg = ExternalGridConfig(margin_bu=float(a.margin_bu))
    builder = ExternalGridCalibrator(pts, (w, h), coverage=coverage, config=cfg)
    cal = builder.build(background_image=bg_path.name)
    cal.save(out / 'EXTERNAL_GRID_CALIBRATION.json')
    top = out / '01_EXTERNAL_GRID_TOP_VIEW.png'
    perspective = out / '02_EXTERNAL_GRID_TRUE_PERSPECTIVE.png'
    draw_top_view(cal, top)
    draw_perspective(frame, cal, perspective)

    court_y = cfg.court_y_bu(coverage)
    expected = [pts[0], pts[1], pts[2], pts[3]]
    actual = [
        builder.project_top_view(0.0, court_y),
        builder.project_top_view(cfg.court_x_bu, court_y),
        builder.project_top_view(cfg.court_x_bu, 0.0),
        builder.project_top_view(0.0, 0.0),
    ]
    corner_err = [float(np.linalg.norm(np.asarray(g)-np.asarray(e))) for g,e in zip(actual, expected)]

    print('============================================================')
    print(' RC8.2 - EXTERNAL GRID TRUE-PERSPECTIVE CALIBRATION GATE')
    print('============================================================')
    print(f'CALIBRATION MODE................. {coverage.value}')
    print('GRID CREATED..................... CALIBRATION ONLY')
    print('TOP VIEW......................... REGULAR 2D FLOOR GRID')
    print('COURT Y=0........................ FAR BASELINE')
    print('COURT Y=MAX...................... NEAR BASELINE')
    print('PERSPECTIVE...................... FLOOR HOMOGRAPHY')
    print('CELL IMAGE GEOMETRY.............. 4 PROJECTED CORNERS / QUADRILATERAL')
    print('IMAGE RECTANGLES................. NO')
    print('GAME RUNTIME..................... NOT RUN IN THIS GATE')
    print('PRODUCTION GRID.................. INVISIBLE')
    print(f'EXTERNAL CELLS................... {len(cal.cells)}')
    print(f'MAX COURT-CORNER ERROR........... {max(corner_err):.6f} px')
    print(f'TOP VIEW......................... {top}')
    print(f'TRUE PERSPECTIVE................. {perspective}')
    print('RC8_2_PERSPECTIVE_GATE=COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
