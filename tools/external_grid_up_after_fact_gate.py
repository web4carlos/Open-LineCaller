from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.external_grid_frame_loop import (
    ExternalGridCalibration,
    ExternalGridFrameLoop,
    LockedBallColorProfile,
)


def draw_poly(img, poly, color, thickness=1):
    pts = np.asarray([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--grid-calibration", required=True)
    ap.add_argument("--start-frame", type=int, default=3822)
    ap.add_argument("--end-frame", type=int, default=3920)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    cal = ExternalGridCalibration.load(a.grid_calibration)
    cal_path = Path(a.grid_calibration)
    bg_path = cal_path.parent / cal.background_image
    bg = Image.open(bg_path).convert("RGB")
    template = Image.open(a.template).convert("RGB")
    profile = LockedBallColorProfile.from_template(template)
    loop = ExternalGridFrameLoop(cal, bg, profile)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("Invalid FPS")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")

    cell_by_id = {c.cell_id: c for c in cal.cells}
    # Engineering display only. Runtime checks every logical cell.
    display_cells = [c for c in cal.cells if (c.ix % 4 == 0 and c.iy % 4 == 0)]
    total = 0
    z0_frames = 0
    bingo_records = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)

    for fn in range(a.start_frame, a.end_frame + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB")
        result = loop.process_frame(fn, rgb)
        total += 1
        if result.z0_candidates:
            z0_frames += 1
        for ev in result.up_confirmations:
            bingo_records.append(ev)

        # Fixed perspective grid: NEVER follows the ball.
        for c in display_cells:
            draw_poly(bgr, c.polygon_image, (70, 70, 70), 1)

        for ev in result.z0_candidates[:6]:
            cell = cell_by_id[ev.cell_id]
            draw_poly(bgr, cell.polygon_image, (255, 255, 255), 2)
            cx, cy = map(int, map(round, ev.centroid_xy))
            cv2.circle(bgr, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(bgr, "Z0 CANDIDATE", (max(5, cx + 6), max(18, cy - 6)), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)

        for ev in result.up_confirmations:
            p0 = tuple(int(round(v)) for v in ev.contact_xy)
            p1 = tuple(int(round(v)) for v in ev.above_xy)
            cv2.arrowedLine(bgr, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=.25)
            cv2.putText(bgr, f"BINGO UP {ev.up_bu:.2f} BU", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)

        status = "BINGO" if result.up_confirmations else ("Z0 CANDIDATE" if result.z0_candidates else "QUIET")
        cv2.putText(bgr, f"FRAME {fn}  {status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, "FRAME -> PILLOW(reference) -> EXTERNAL GRID -> Z0 -> UP AFTER FACT -> BINGO", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(bgr)

    writer.release()
    cap.release()

    print("============================================================")
    print(" CP-0035.9.4.2 RC8.2 - EXTERNAL GRID + CALIBRATED REFERENCE + UP")
    print("============================================================")
    print(f"FPS.............................. {fps:.6f}")
    print(f"CALIBRATION MODE................. {cal.coverage}")
    print("GRID BUILD AT RUNTIME............ NO")
    print("BALL TRACKER...................... NO")
    print("PATH / XYZ........................ NO")
    print("DOWN REQUIREMENT.................. NO")
    print("WATCHED AREA...................... EXTERNAL GRID ONLY")
    print("PILLOW REFERENCE................... CALIBRATED CELL BACKGROUND")
    print("PREVIOUS FRAME REFERENCE........... NO")
    print("RUNTIME ORDER..................... FRAME -> PILLOW(reference) -> CELL LOOP -> Z0 -> UP -> BINGO")
    print(f"EXTERNAL LOGICAL CELLS............ {len(cal.cells)}")
    print(f"FRAMES............................ {total}")
    print(f"FRAMES WITH Z0 CANDIDATE.......... {z0_frames}")
    print(f"UP-CONFIRMED BINGOS............... {len(bingo_records)}")
    print()
    print("BINGO RECORDS:")
    if bingo_records:
        for ev in bingo_records[:40]:
            print(
                f"contact_frame={ev.contact_frame} confirm_frame={ev.confirm_frame} "
                f"cell={ev.cell_id} region={ev.region} up={ev.up_bu:.3f}BU "
                f"lateral={ev.lateral_bu:.3f}BU scale_after={ev.scale_ratio_after:.3f}"
            )
        if len(bingo_records) > 40:
            print(f"... {len(bingo_records) - 40} more")
    else:
        print("NONE")
    print()
    print(f"output={out}")
    print("GRID OVERLAY....................... ENGINEERING ONLY")
    print("PRODUCTION GRID.................... INVISIBLE")
    print("OFFICIAL GAME OUT................. NOT ENABLED BY THIS RC")
    print("RC8_2_EXTERNAL_GRID_UP_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
