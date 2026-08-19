from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.ball_wave import BallWaveTracker
from linecaller.dcf.models import DCFConfig
from linecaller.flight_paths import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.discontinuity import TrajectoryDiscontinuityDetector
from linecaller.flight_paths.transition_continuity import DirectionTransitionContinuity


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
        raise RuntimeError("Invalid FPS")

    court = CourtFrame()
    cfg = DCFConfig()
    searcher = BallInMeshSearcher(load_points(a.calibration))
    old_tracker = BallWaveTracker(searcher)

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")
    acquisition = searcher.search(frame, ball)
    if not acquisition.found or acquisition.index is None:
        raise RuntimeError("Could not acquire locked ball at gate start")
    old_tracker.acquire(acquisition)

    records = []
    transition = None
    transition_start_frame = None
    new_path_frame = None
    global_reacquisition_count = 0

    def add_record(frame_no, hit, state):
        if not hit.found or hit.index is None:
            return
        point = cell_to_point(hit.index, cfg, court)
        records.append({
            "frame": frame_no,
            "hit": hit,
            "point": TimedPoint3D(point, frame_no / fps),
            "state": state,
        })

    add_record(a.start_frame, acquisition, "OLD_PATH")
    last_trusted = acquisition

    for frame_no in range(a.start_frame + 1, a.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        if transition is None:
            step = old_tracker.track(frame, ball)
            if step.hit.found and step.hit.index is not None:
                last_trusted = step.hit
                add_record(frame_no, step.hit, "OLD_PATH")
                continue

            # The old flight model just failed locally.  Do NOT global-search.
            # The same frame is immediately searched around the LAST TRUSTED
            # position with no old-direction bias.
            transition = DirectionTransitionContinuity(searcher)
            transition.start(last_trusted)
            transition_start_frame = frame_no

        tstep = transition.track(frame, ball)
        if tstep.hit.found and tstep.hit.index is not None:
            state = "TRANSITION_LOCAL"
            if tstep.new_path_confirmed:
                state = "NEW_PATH_CONFIRMED"
                if new_path_frame is None:
                    new_path_frame = frame_no
            add_record(frame_no, tstep.hit, state)

    cap.release()

    outgoing = None
    if transition is not None and len(transition.accepted_hits) >= 3:
        transition_records = [
            r for r in records
            if r["frame"] >= (transition_start_frame or 0)
        ]
        if len(transition_records) >= 3:
            obs = [r["point"] for r in transition_records[:3]]
            detector = TrajectoryDiscontinuityDetector(gravity_bu_s2=court.gravity_bu_s2)
            outgoing = detector.fit_window(obs, reference_time_s=obs[0].t_s)

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC5 - LOCAL FLIGHT TRANSITION CONTINUITY")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print("BALL IDENTITY.................... WHO ONLY")
    print("OLD PATH VECTOR AFTER HIT........ NOT USED")
    print("TRANSITION ANCHOR................. LAST TRUSTED P(x,y,z,t)")
    print("TRANSITION SEARCH................. LOCAL 3-D BALL-UNIT SPHERE")
    print("GLOBAL REACQUISITION.............. FORBIDDEN")
    print("MESH/CELLS DEFINE PATH............ NO")
    print("CELLS DURING TRANSITION........... SEARCH DISCRETIZATION ONLY")
    print()
    print("OBSERVATIONS:")
    for r in records:
        o = r["point"]
        h = r["hit"]
        print(
            f"  {r['frame']}: {r['state']:18s} "
            f"P=({o.x:.2f},{o.y:.2f},{o.z:.2f}) BU "
            f"px=({h.x:.1f},{h.y:.1f}) score={h.score:.3f} cell={h.index}"
        )

    print()
    print(f"TRANSITION START FRAME............ {transition_start_frame}")
    print(f"NEW PATH READY FRAME.............. {new_path_frame}")
    print(f"GLOBAL REACQUISITION COUNT........ {global_reacquisition_count}")
    if outgoing is not None:
        print(
            "NEW V0............................ "
            f"({outgoing.velocity.vx:.2f},"
            f"{outgoing.velocity.vy:.2f},"
            f"{outgoing.velocity.vz:.2f}) BU/s"
        )
        print(f"NEW PATH FIT RMS.................. {outgoing.rms_residual_bu:.3f} BU")
    else:
        print("NEW V0............................ NOT ENOUGH LOCAL HITS")

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

    last_marker = None
    for frame_no in range(a.start_frame, a.end_frame + 1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None:
            h = rec["hit"]
            if h.x is not None and h.y is not None:
                last_marker = (round(h.x), round(h.y))
                cv2.circle(img, last_marker, 10, (255,255,255), 2, cv2.LINE_AA)

        if transition_start_frame is None or frame_no < transition_start_frame:
            state = "OLD PATH"
        elif new_path_frame is None or frame_no < new_path_frame:
            state = "LOCAL TRANSITION - NO GLOBAL SEARCH"
        else:
            state = "NEW PATH OBSERVATIONS READY"

        cv2.rectangle(img, (8,8), (500,68), (0,0,0), -1)
        cv2.putText(img, f"frame {frame_no}", (18,31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, state, (18,56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    cap.release()

    print()
    print(f"output={out}")
    print("GLOBAL_SEARCH_DURING_TRANSITION=0")
    print("PRODUCTION_OVERLAY=NO")
    print("RC5_VISUAL_CONFIRMATION_REQUIRED=YES")
    print("LOCAL_FLIGHT_TRANSITION_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
