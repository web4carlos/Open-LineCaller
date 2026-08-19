from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from linecaller.dcf.external_grid_pillow_watcher import ExternalGridCalibration


def read_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index}")
    return frame


def load_court_calibration(path: str):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    points = d.get("image_points")
    if points is None:
        raise RuntimeError("Calibration JSON needs image_points.")
    mode = d.get("mode", "FULL_COURT")
    return mode, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--court-calibration", required=True)
    ap.add_argument("--background-frame", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--margin-bu", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = read_frame(args.video, args.background_frame)
    h, w = frame.shape[:2]
    mode, points = load_court_calibration(args.court_calibration)

    reference_name = "EXTERNAL_GRID_REFERENCE.png"
    reference_path = out / reference_name
    cv2.imwrite(str(reference_path), frame)

    cal = ExternalGridCalibration.build(
        image_size=(w, h),
        mode=mode,
        image_points=points,
        margin_bu=args.margin_bu,
        reference_image=reference_name,
    )
    json_path = out / "EXTERNAL_GRID_PILLOW_CALIBRATION.json"
    cal.save(json_path)

    print("")
    print("=" * 60)
    print(" RC8.3 - EXTERNAL GRID PILLOW CALIBRATION")
    print("=" * 60)
    print(f"OWNER............................ COURT CALIBRATION")
    print(f"MODE............................. {mode}")
    print(f"REFERENCE FRAME.................. {args.background_frame}")
    print(f"PILLOW REFERENCE................. CALIBRATED BACKGROUND")
    print(f"PREVIOUS FRAME................... NO")
    print(f"EXTERNAL VISIBLE CELLS........... {len(cal.cells)}")
    print(f"MARGIN........................... {args.margin_bu} BU")
    print(f"CALIBRATION...................... {json_path}")
    print(f"REFERENCE........................ {reference_path}")
    print("GAME RUNTIME..................... NOT RUN")
    print("")


if __name__ == "__main__":
    main()
