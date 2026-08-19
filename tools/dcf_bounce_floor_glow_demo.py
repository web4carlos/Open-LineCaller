from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.ball_wave import BallWaveTracker
from linecaller.dcf.bounce_floor_glow import BounceFloorGlowController


def load_image_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("image_points")
    if not isinstance(pts, list) or len(pts) != 4:
        raise ValueError("calibration must contain four image_points")
    return np.asarray(pts, dtype=np.float64)


def parse_color_bgr(text: str):
    values = tuple(int(v.strip()) for v in text.split(","))
    if len(values) != 3 or any(v < 0 or v > 255 for v in values):
        raise ValueError("--color-bgr must be B,G,R values in [0,255]")
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--output", required=True)
    p.add_argument("--color-bgr", default="0,210,255")
    p.add_argument("--auto-reacquire", action="store_true")
    args = p.parse_args()

    color_bgr = parse_color_bgr(args.color_bgr)
    ball = cv2.imread(args.template)
    if ball is None:
        raise RuntimeError(f"Could not read ball template: {args.template}")

    searcher = BallInMeshSearcher(load_image_points(args.calibration))
    wave = BallWaveTracker(searcher)
    glow = BounceFloorGlowController(searcher.projector)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output: {out_path}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read acquisition frame")

    acquisition = searcher.search(frame, ball)
    if not acquisition.found or acquisition.index is None:
        raise RuntimeError("One-time acquisition failed")
    wave.acquire(acquisition)

    frame_number = int(args.start_frame)
    writer.write(frame)
    glow_events = []

    for _ in range(max(0, int(args.frames) - 1)):
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        now_s = frame_number / fps

        step = wave.track(frame, ball)
        if step.state.requires_reacquisition and args.auto_reacquire:
            step = wave.reacquire(frame, ball)

        event = glow.consume_wave_step(step, now_s=now_s, color_bgr=color_bgr)
        if event is not None:
            glow_events.append((frame_number, event.cell))
            print(f"FLOOR_GLOW frame={frame_number} cell={event.cell}")

        # Production visual contract: video + glow only.
        writer.write(glow.render(frame, now_s=now_s))

    cap.release()
    writer.release()

    print("")
    print("CP-0035.9.5 BOUNCE FLOOR GLOW")
    print(f"glow_events={glow_events}")
    print("trigger_source=CP-0035.9.4.1 floor_trigger ONLY")
    print("Z1_DOWN_PREDICTED_Z0=SUPPORTED")
    print("DIRECT_DESCENDING_Z0=SUPPORTED")
    print("MESH_VISIBLE=NO")
    print("BALL_WAVE_VISIBLE=NO")
    print("CELL_COORDINATES_VISIBLE=NO")
    print("DECISION_COLOR_MAPPING=NOT_IN_THIS_CP")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
