from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.ball_wave import BallWaveTracker


def load_image_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("image_points")
    if not isinstance(pts, list) or len(pts) != 4:
        raise ValueError(
            "Calibration must contain 4 image_points in order: "
            "near-left, near-right, far-right, far-left."
        )
    return np.asarray(pts, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--output", default="")
    p.add_argument("--debug-overlay", action="store_true")
    p.add_argument("--auto-reacquire", action="store_true")
    args = p.parse_args()

    ball = cv2.imread(args.template)
    if ball is None:
        raise RuntimeError(f"Could not read ball template: {args.template}")

    image_points = load_image_points(args.calibration)
    searcher = BallInMeshSearcher(image_points)
    wave = BallWaveTracker(searcher)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    writer = None
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

    first_ok, first = cap.read()
    if not first_ok:
        raise RuntimeError("Could not read acquisition frame")

    acquisition = searcher.search(first, ball)
    if not acquisition.found:
        print("acquisition_found=False")
        cap.release()
        if writer:
            writer.release()
        raise SystemExit(2)

    wave.acquire(acquisition)

    local_hits = 0
    local_misses = 0
    reacquisitions = 0
    contact_candidates = 1 if acquisition.index and acquisition.index.z == 0 else 0
    max_wave_cells = wave.state.wave_cell_count
    local_candidate_sum = 0
    local_frames = 0

    def render(frame, hit, label):
        if not args.debug_overlay:
            return frame
        vis = frame.copy()
        if hit.found and hit.x is not None and hit.y is not None:
            cv2.circle(
                vis,
                (round(hit.x), round(hit.y)),
                10,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if hit.index is not None:
                cv2.putText(
                    vis,
                    f"{label} DCF {hit.index.x},{hit.index.y},{hit.index.z}",
                    (round(hit.x) + 12, round(hit.y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return vis

    if writer:
        writer.write(render(first, acquisition, "ACQUIRE"))

    for _ in range(max(0, args.frames - 1)):
        ok, frame = cap.read()
        if not ok:
            break

        step = wave.track(frame, ball)
        local_frames += 1
        local_candidate_sum += step.local_candidates_tested
        max_wave_cells = max(max_wave_cells, step.state.wave_cell_count)

        if step.hit.found:
            local_hits += 1
        else:
            local_misses += 1

        if step.contact_plane_reached:
            contact_candidates += 1

        if step.state.requires_reacquisition and args.auto_reacquire:
            step = wave.reacquire(frame, ball)
            if step.hit.found:
                reacquisitions += 1

        if writer:
            writer.write(render(frame, step.hit, step.reason))

    cap.release()
    if writer:
        writer.release()

    avg_candidates = (
        local_candidate_sum / local_frames
        if local_frames
        else 0.0
    )

    print("CP-0035.9.4 - DCF BALL WAVE")
    print("SPORT=pickleball-primary")
    print("YOLO=OFF")
    print("PRODUCTION_MESH_VISIBLE=NO")
    print("FLOOR_ILLUMINATION=NO")
    print(f"acquisition_found={acquisition.found}")
    print(
        "acquisition_cell="
        f"{None if acquisition.index is None else (acquisition.index.x, acquisition.index.y, acquisition.index.z)}"
    )
    print(f"local_hits={local_hits}")
    print(f"local_misses={local_misses}")
    print(f"reacquisitions={reacquisitions}")
    print(f"contact_plane_candidates={contact_candidates}")
    print(f"max_wave_cells={max_wave_cells}")
    print(f"avg_local_candidates={avg_candidates:.2f}")
    print(f"final_state={wave.state}")


if __name__ == "__main__":
    main()
