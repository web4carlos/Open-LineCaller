from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.ball_wave import BallWaveTracker
from linecaller.dcf.models import DCFConfig
from linecaller.flight_paths import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.discontinuity import (
    DirectionChangeConfig,
    TrajectoryDiscontinuityDetector,
    TransitionKind,
)


def load_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["image_points"], dtype=np.float64)


def cell_to_point(cell, cfg: DCFConfig, court: CourtFrame) -> Point3D:
    sx = court.width_bu / cfg.court_x_cells
    sy = court.length_bu / cfg.court_y_cells
    return Point3D(
        (cell.x - cfg.margin_cells + 0.5) * sx,
        (cell.y - cfg.margin_cells + 0.5) * sy,
        float(cell.z),
    )


def fmt_v(v):
    return f"({v.vx:.2f},{v.vy:.2f},{v.vz:.2f}) BU/s"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--start-frame", type=int, default=3808)
    p.add_argument("--end-frame", type=int, default=3822)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    ball = cv2.imread(a.template)
    if ball is None:
        raise RuntimeError("Cannot read ball reference")
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Invalid video FPS")

    court = CourtFrame()
    cfg = DCFConfig()
    searcher = BallInMeshSearcher(load_points(a.calibration))
    tracker = BallWaveTracker(searcher)

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")
    acq = searcher.search(frame, ball)
    if not acq.found or acq.index is None:
        raise RuntimeError("Could not acquire locked ball at start frame")
    tracker.acquire(acq)

    records = []
    def record(frame_no, hit, reason):
        if not hit.found or hit.index is None:
            return
        pt = cell_to_point(hit.index, cfg, court)
        records.append({
            "frame": frame_no,
            "obs": TimedPoint3D(pt, frame_no / fps),
            "cell": hit.index,
            "px": hit.x,
            "py": hit.y,
            "reason": reason,
        })

    record(a.start_frame, acq, "ACQUIRED")

    for frame_no in range(a.start_frame + 1, a.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        step = tracker.track(frame, ball)
        if step.hit.found:
            record(frame_no, step.hit, step.reason)
        elif step.state.requires_reacquisition:
            # ENGINEERING GATE ONLY. Production path-following policy remains a
            # separate decision. Here we need post-change observations to prove
            # OLD FLIGHT -> discontinuity -> NEW FLIGHT on the same locked ball.
            reacq = tracker.reacquire(frame, ball)
            if reacq.hit.found:
                record(frame_no, reacq.hit, reacq.reason)
    cap.release()

    detector = TrajectoryDiscontinuityDetector(
        DirectionChangeConfig(
            max_internal_gap_s=max(0.075, 1.8 / fps),
            max_transition_gap_s=max(0.250, 7.0 / fps),
        ),
        gravity_bu_s2=court.gravity_bu_s2,
    )
    candidates = detector.scan([r["obs"] for r in records])
    new_flight = next((c for c in candidates if c.is_new_flight_candidate), None)

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC4 - REAL FLIGHT-EPOCH DIRECTION GATE")
    print("============================================================")
    print()
    print(f"FPS.......................... {fps:.6f}")
    print("BALL IDENTITY................ WHO ONLY")
    print("PATH DATA.................... P(x,y,z,t)")
    print("CELLS DEFINE PATH............ NO")
    print("DIRECTION CHANGE.............. NEW FLIGHT CANDIDATE")
    print("FLOOR BOUNCE.................. CLASSIFIED SEPARATELY")
    print("REACQUISITION................. ENGINEERING GATE ONLY")
    print()
    print("REAL OBSERVATIONS:")
    for r in records:
        o = r["obs"]
        print(
            f"  {r['frame']}: P=({o.x:.2f},{o.y:.2f},{o.z:.2f}) BU "
            f"t={o.t_s:.6f}s cell={r['cell']} {r['reason']}"
        )

    print()
    print("TRANSITION CANDIDATES:")
    if not candidates:
        print("  NONE")
    for i, c in enumerate(candidates, 1):
        f0 = round(c.transition_start_s * fps)
        f1 = round(c.transition_end_s * fps)
        print(
            f"  #{i} {c.kind.value} frames={f0}..{f1} "
            f"angle={c.angle_deg:.2f}deg residual={c.old_path_residual_bu:.2f}BU "
            f"speed_change={c.speed_change_ratio:.3f} score={c.score:.3f}"
        )

    if new_flight is not None:
        change_start = round(new_flight.transition_start_s * fps)
        change_end = round(new_flight.transition_end_s * fps)
        print()
        print("STRONGEST NEW-FLIGHT CANDIDATE")
        print(f"WINDOW....................... {change_start}..{change_end}")
        print(f"INCOMING V................... {fmt_v(new_flight.incoming.velocity)}")
        print(f"OUTGOING V................... {fmt_v(new_flight.outgoing.velocity)}")
        print(f"ANGLE CHANGE................. {new_flight.angle_deg:.2f} deg")
        print(f"OLD PATH RESIDUAL............ {new_flight.old_path_residual_bu:.2f} BU")
        print("STATUS........................ VISUAL CONFIRMATION REQUIRED")
    else:
        change_start = change_end = None
        print()
        print("STRONGEST NEW-FLIGHT CANDIDATE.... NONE")

    # Engineering video: actual locked-ball observation marker + transition
    # window. No mesh, no candidate cells, no production overlay semantics.
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_frame = {r["frame"]: r for r in records}
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")

    for frame_no in range(a.start_frame, a.end_frame + 1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None and rec["px"] is not None and rec["py"] is not None:
            cv2.circle(img, (round(rec["px"]), round(rec["py"])), 10, (255,255,255), 2, cv2.LINE_AA)

        if change_start is None:
            state = "NO NEW-FLIGHT CANDIDATE"
        elif frame_no <= change_start:
            state = "OLD FLIGHT"
        elif frame_no <= change_end:
            state = "DIRECTION-CHANGE WINDOW"
        else:
            state = "NEW FLIGHT"

        cv2.rectangle(img, (8, 8), (420, 68), (0,0,0), -1)
        cv2.putText(img, f"frame {frame_no}", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, state, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    cap.release()

    print()
    print(f"output={out}")
    print("MESH_VISIBLE=NO")
    print("PRODUCTION_OVERLAY=NO")
    print("REAL_DIRECTION_CHANGE_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
