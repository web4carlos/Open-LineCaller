from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher


def read_frame(video: str, frame_no: int):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_no}")
    return frame


def load_image_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("image_points")
    if not isinstance(pts, list) or len(pts) != 4:
        raise ValueError(
            "Calibration must contain image_points in order: "
            "near-left, near-right, far-right, far-left."
        )
    return np.asarray(pts, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", required=True, type=int)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    frame = read_frame(args.video, args.frame)
    ball = cv2.imread(args.template)
    if ball is None:
        raise RuntimeError(f"Could not read template: {args.template}")

    image_points = load_image_points(args.calibration)
    searcher = BallInMeshSearcher(image_points)
    hit = searcher.search(frame, ball)

    # Production visualization rule:
    # NO MESH. Only final useful evidence.
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
        if args.debug and hit.index is not None:
            cv2.putText(
                vis,
                f"DCF {hit.index.x},{hit.index.y},{hit.index.z}",
                (round(hit.x) + 14, round(hit.y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / f"frame_{args.frame}_dcf_ball_in_mesh.jpg"
    cv2.imwrite(str(image_path), vis)

    identity = searcher.ball_identity

    print("CP-0035.9.3 RC2 - RUNTIME DCF BALL-IN-MESH")
    print("SPORT=pickleball-primary")
    print("YOLO=OFF")
    print("PRODUCTION_MESH_VISIBLE=NO")
    print(f"ball_locked={searcher.ball_locked}")
    print(f"ball_generation={searcher.ball_generation}")
    print(f"reference_ball_diameter_px={None if identity is None else identity.diameter_px:.4f}")
    print(f"ball_found={hit.found}")
    print(f"score={hit.score:.6f}")
    print(f"appearance_score={hit.template_score:.6f}")
    print(f"color_score={hit.color_score:.6f}")
    print(f"center_score={hit.center_score:.6f}")
    print(f"offset_px={hit.offset_px}")
    print(f"x={hit.x}")
    print(f"y={hit.y}")
    print(f"dcf_cell={None if hit.index is None else (hit.index.x, hit.index.y, hit.index.z)}")
    print(f"region={None if hit.region is None else hit.region.value}")
    print(f"coarse_candidates_tested={hit.coarse_candidates_tested}")
    print(f"refined_candidates_tested={hit.refined_candidates_tested}")
    print(f"image={image_path}")


if __name__ == "__main__":
    main()
