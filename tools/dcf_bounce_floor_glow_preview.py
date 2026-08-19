from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.bounce_floor_glow import BounceFloorGlowController
from linecaller.dcf.models import CellIndex


def load_points(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["image_points"], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--frame", type=int, default=3808)
    p.add_argument("--output", required=True)
    p.add_argument("--color-bgr", default="0,210,255")
    args = p.parse_args()

    color = tuple(int(v) for v in args.color_bgr.split(","))
    ball = cv2.imread(args.template)
    if ball is None:
        raise RuntimeError("Cannot read ball template")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read preview frame")

    searcher = BallInMeshSearcher(load_points(args.calibration))
    hit = searcher.search(frame, ball)
    if not hit.found or hit.index is None:
        raise RuntimeError("Preview acquisition failed")

    # STYLE PREVIEW ONLY: same acquired X/Y projected to Z=0. This is explicitly
    # not treated as a detected bounce and never enters production logic.
    contact = CellIndex(hit.index.x, hit.index.y, 0)
    glow = BounceFloorGlowController(searcher.projector)
    glow.trigger_contact(contact, now_s=0.0, color_bgr=color, ball_generation=hit.ball_generation)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open preview output")

    total = int(round(glow.config.duration_ms / 1000.0 * fps)) + 10
    for i in range(total):
        now_s = i / fps
        vis = glow.render(frame, now_s=now_s)
        cv2.putText(
            vis,
            "STYLE PREVIEW ONLY - INJECTED Z=0 - NOT A DETECTED BOUNCE",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(vis)
    writer.release()

    print(f"acquired_cell={hit.index}")
    print(f"preview_floor_cell={contact}")
    print("detected_bounce=NO")
    print("production_mesh_visible=NO")
    print(f"output={output}")


if __name__ == "__main__":
    main()
