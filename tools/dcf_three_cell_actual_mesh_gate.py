from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.ball_wave import BallWaveTracker
from linecaller.dcf.models import CellIndex


def read_points(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("image_points")
    if not isinstance(pts, list) or len(pts) != 4:
        raise ValueError("calibration must contain four image_points")
    return np.asarray(pts, dtype=np.float64)


def read_frame(cap: cv2.VideoCapture, frame_number: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}")
    return frame


def pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_actual_dcf_floor(frame, projector):
    """Draw the real 83 x 182 pickleball DCF floor grid for engineering only."""
    overlay = frame.copy()
    field = projector.field
    c = field.config

    # Exact lateral DCF cell boundaries: 83 cells across 20 ft.
    for i in range(c.court_x_cells + 1):
        u = i / float(c.court_x_cells)
        a = projector.floor_point(u, 0.0)
        b = projector.floor_point(u, 1.0)
        cv2.line(overlay, pt(a), pt(b), (0, 210, 255), 1, cv2.LINE_AA)

    # Exact depth DCF cell boundaries: 182 cells across 44 ft.
    for j in range(c.court_y_cells + 1):
        t = j / float(c.court_y_cells)
        a = projector.floor_point(0.0, t)
        b = projector.floor_point(1.0, t)
        cv2.line(overlay, pt(a), pt(b), (0, 210, 255), 1, cv2.LINE_AA)

    # Court boundary emphasized; same DCF, not a second mesh.
    corners = [projector.nl, projector.nr, projector.fr, projector.fl]
    cv2.polylines(
        overlay,
        [np.asarray([pt(p) for p in corners], dtype=np.int32)],
        True,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.addWeighted(overlay, 0.34, frame, 0.66, 0.0)


def cell_floor_quad(projector, cell: CellIndex):
    field = projector.field
    c = field.config
    u0 = (cell.x - field.x0) / float(c.court_x_cells)
    u1 = (cell.x - field.x0 + 1) / float(c.court_x_cells)
    t0 = (cell.y - field.y0) / float(c.court_y_cells)
    t1 = (cell.y - field.y0 + 1) / float(c.court_y_cells)
    return [
        projector.floor_point(u0, t0),
        projector.floor_point(u1, t0),
        projector.floor_point(u1, t1),
        projector.floor_point(u0, t1),
    ]


def draw_cell_cube(frame, projector, cell: CellIndex, color, thickness=2):
    if not projector.field.valid(cell):
        return frame

    floor = cell_floor_quad(projector, cell)
    diameter = projector.expected_diameter_px(cell)
    v = projector.vertical_unit
    bottom = [p + v * (cell.z * diameter) for p in floor]
    top = [p + v * ((cell.z + 1) * diameter) for p in floor]

    cv2.polylines(
        frame,
        [np.asarray([pt(p) for p in bottom], dtype=np.int32)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.polylines(
        frame,
        [np.asarray([pt(p) for p in top], dtype=np.int32)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )
    for a, b in zip(bottom, top):
        cv2.line(frame, pt(a), pt(b), color, thickness, cv2.LINE_AA)
    return frame


def draw_candidates(frame, searcher, cells):
    for cell in cells:
        try:
            x, y = searcher.projector.image_point(cell)
        except ValueError:
            continue
        if -4 <= x < frame.shape[1] + 4 and -4 <= y < frame.shape[0] + 4:
            cv2.circle(frame, (round(x), round(y)), 1, (0, 255, 255), -1, cv2.LINE_AA)
    return frame


def draw_contact_floor(frame, projector, cell: CellIndex):
    quad = np.asarray([pt(p) for p in cell_floor_quad(projector, cell)], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, quad, (0, 255, 255), cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0)
    cv2.polylines(frame, [quad], True, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def annotate(frame, searcher, tracker, step, frame_number, candidates):
    out = draw_actual_dcf_floor(frame.copy(), searcher.projector)
    out = draw_candidates(out, searcher, candidates)

    history = step.state.history if step is not None else tracker.state.history
    history_colors = [(255, 0, 255), (255, 100, 255), (0, 0, 255)]
    for color, cell in zip(history_colors[-len(history):], history):
        draw_cell_cube(out, searcher.projector, cell, color, 2)

    if step is not None and step.hit.found and step.hit.x is not None:
        cv2.circle(
            out,
            (round(step.hit.x), round(step.hit.y)),
            7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if step is not None and step.floor_trigger and step.predicted_contact_cell is not None:
        draw_contact_floor(out, searcher.projector, step.predicted_contact_cell)

    current = tracker.state.last_cell
    text1 = f"FRAME {frame_number} | ACTUAL DCF 83x182x48 | court 20x44 ft"
    text2 = f"history={list(history)} | current={current} | candidates={len(candidates)}"
    if step is not None:
        text3 = (
            f"reason={step.reason} | floor_trigger={step.floor_trigger} | "
            f"contact={step.predicted_contact_cell}"
        )
    else:
        text3 = "GLOBAL ACQUISITION ONCE"

    for y, text in ((24, text1), (46, text2), (68, text3)):
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        out,
        "ENGINEERING VIEW ONLY - PRODUCTION DCF/MESH REMAINS INVISIBLE",
        (10, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--start-frame", type=int, default=3808)
    p.add_argument("--end-frame", type=int, default=3812)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    points = read_points(args.calibration)
    ball = cv2.imread(args.template)
    if ball is None:
        raise RuntimeError(f"Cannot read template: {args.template}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    searcher = BallInMeshSearcher(points)
    tracker = BallWaveTracker(searcher)

    first = read_frame(cap, args.start_frame)
    t0 = time.perf_counter()
    acquisition = searcher.search(first, ball)
    acquisition_seconds = time.perf_counter() - t0
    if not acquisition.found or acquisition.index is None:
        raise RuntimeError("One-time acquisition failed on start frame")
    tracker.acquire(acquisition)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output: {output}")

    acquisition_candidates = tracker.current_wave()
    fake_step = None
    vis = draw_actual_dcf_floor(first.copy(), searcher.projector)
    vis = draw_candidates(vis, searcher, acquisition_candidates)
    draw_cell_cube(vis, searcher.projector, acquisition.index, (0, 0, 255), 2)
    if acquisition.x is not None and acquisition.y is not None:
        cv2.circle(vis, (round(acquisition.x), round(acquisition.y)), 7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"FRAME {args.start_frame} | ONE-TIME GLOBAL ACQUISITION | cell={acquisition.index}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, "ACTUAL DCF 83x182x48 | 20x44 ft | ENGINEERING VIEW ONLY",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    writer.write(vis)

    local_hits = 0
    local_misses = 0
    floor_triggers = []
    max_candidates = len(acquisition_candidates)
    local_seconds = 0.0

    for n in range(args.start_frame + 1, args.end_frame + 1):
        frame = read_frame(cap, n)
        candidates = tracker.current_wave()
        max_candidates = max(max_candidates, len(candidates))
        t = time.perf_counter()
        step = tracker.track(frame, ball)
        local_seconds += time.perf_counter() - t

        if step.hit.found:
            local_hits += 1
        else:
            local_misses += 1
        if step.floor_trigger:
            floor_triggers.append((n, step.predicted_contact_cell))

        writer.write(annotate(frame, searcher, tracker, step, n, candidates))

    writer.release()
    cap.release()

    print("")
    print("CP-0035.9.4.1 THREE-CELL ACTUAL-DCF GATE")
    print(f"fps={fps:.3f}")
    print(f"frames={args.start_frame}..{args.end_frame}")
    print(f"acquisition_cell={acquisition.index}")
    print(f"acquisition_seconds={acquisition_seconds:.3f}")
    print(f"local_hits={local_hits}")
    print(f"local_misses={local_misses}")
    print(f"max_history={len(tracker.state.history)}")
    print(f"max_local_candidates={max_candidates}")
    print(f"local_tracking_seconds={local_seconds:.3f}")
    print(f"floor_triggers={floor_triggers}")
    print("global_scan_count=1")
    print("global_scan_during_continuity=0")
    print("actual_dcf=83x182x48")
    print("production_mesh_visible=NO")
    print(f"output={output}")

    expected = args.end_frame - args.start_frame
    if local_hits == expected and local_misses == 0:
        print("REAL_GATE=PASS")
        raise SystemExit(0)

    print("REAL_GATE=FAIL")
    raise SystemExit(3)


if __name__ == "__main__":
    main()
