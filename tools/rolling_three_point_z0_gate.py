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
from linecaller.flight_paths.rolling_z0 import RollingZ0Tracker
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


def floor_to_image(point: Point3D, image_points, court: CourtFrame):
    # Continuous floor projection.  This is for engineering visualization only;
    # it does not define or alter the rolling Z0 mathematics.
    q = np.asarray(image_points, dtype=np.float64)
    nl, nr, fr, fl = q
    u = point.x / court.width_bu
    t = point.y / court.length_bu
    left = nl * (1.0 - t) + fl * t
    right = nr * (1.0 - t) + fr * t
    p = left * (1.0 - u) + right * u
    return float(p[0]), float(p[1])


def distance_xy(a: Point3D, b: Point3D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--start-frame", type=int, default=3808)
    ap.add_argument("--end-frame", type=int, default=3822)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    ball = cv2.imread(a.template)
    if ball is None:
        raise RuntimeError("Cannot read ball reference")
    image_points = load_points(a.calibration)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Invalid FPS")

    court = CourtFrame()
    cfg = DCFConfig()
    searcher = BallInMeshSearcher(image_points)
    old_tracker = BallWaveTracker(searcher)
    rolling = RollingZ0Tracker(gravity_bu_s2=court.gravity_bu_s2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")
    acquisition = searcher.search(frame, ball)
    if not acquisition.found or acquisition.index is None:
        raise RuntimeError("Could not acquire locked ball at gate start")
    old_tracker.acquire(acquisition)

    records = []
    predictions = []
    transition = None
    transition_start_frame = None
    global_reacquisition_count = 0

    def add_record(frame_no, hit, state, feed_rolling=False):
        if not hit.found or hit.index is None:
            return None
        obs = TimedPoint3D(cell_to_point(hit.index, cfg, court), frame_no / fps)
        rec = {
            "frame": frame_no,
            "hit": hit,
            "point": obs,
            "state": state,
        }
        records.append(rec)
        if feed_rolling:
            pred = rolling.add(obs)
            if pred is not None:
                predictions.append({"frame": frame_no, "prediction": pred})
                rec["prediction"] = pred
        return rec

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

            # Old local flight continuity failed.  Same BallSession continues,
            # but the old direction is no longer trusted.  No global search.
            transition = DirectionTransitionContinuity(searcher)
            transition.start(last_trusted)
            transition_start_frame = frame_no
            rolling.reset()

        tstep = transition.track(frame, ball)
        if tstep.hit.found and tstep.hit.index is not None:
            state = "ROLLING_1_OF_3"
            n = len(transition.accepted_hits)
            if n == 2:
                state = "ROLLING_2_OF_3"
            elif n >= 3:
                state = "ROLLING_Z0_ACTIVE"
            add_record(frame_no, tstep.hit, state, feed_rolling=True)

    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC6 - ROLLING 3-POINT Z0 DIAGNOSTIC")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print("BALL IDENTITY.................... WHO ONLY")
    print("ROLLING WINDOW................... MAX 3 P(x,y,z,t)")
    print("COMPLETE PATH STORED............. NO")
    print("EVERY NEW POINT.................. DROP OLDEST / RECOMPUTE Z0")
    print("GLOBAL REACQUISITION.............. FORBIDDEN")
    print("MESH/CELLS DEFINE Z0 MATH......... NO")
    print("CELLS............................. OBSERVATION DISCRETIZATION ONLY")
    print()
    print(f"TRANSITION START FRAME............ {transition_start_frame}")
    print(f"GLOBAL REACQUISITION COUNT........ {global_reacquisition_count}")
    print()
    print("ROLLING Z0 ESTIMATES:")
    print("FRAME | WINDOW        | CURRENT P                  | Vcurrent                   | dt->Z0 | predicted Z0")
    print("-" * 132)
    prev_landing = None
    for item in predictions:
        pred = item["prediction"]
        frames = [round(o.t_s * fps) for o in pred.observations]
        shift = None if prev_landing is None else distance_xy(pred.landing, prev_landing)
        shift_txt = "" if shift is None else f" shift={shift:.2f}BU"
        print(
            f"{item['frame']:5d} | "
            f"{frames[0]}-{frames[1]}-{frames[2]} | "
            f"({pred.current.x:6.2f},{pred.current.y:6.2f},{pred.current.z:5.2f}) | "
            f"({pred.velocity.vx:7.2f},{pred.velocity.vy:7.2f},{pred.velocity.vz:7.2f}) | "
            f"{pred.time_to_z0_s:6.3f} | "
            f"({pred.landing.x:7.2f},{pred.landing.y:7.2f},0){shift_txt}"
        )
        prev_landing = pred.landing

    print()
    if predictions:
        last = predictions[-1]["prediction"]
        print(
            "LATEST Z0....................... "
            f"({last.landing.x:.2f},{last.landing.y:.2f},0.00) BU"
        )
        print(f"LATEST TIME TO Z0................. {last.time_to_z0_s:.4f} s")
        print(f"LATEST CONTACT TIME................ {last.contact_time_s:.6f} s")
    else:
        print("LATEST Z0......................... NOT ENOUGH 3-POINT WINDOWS")

    print()
    print("IMPORTANT: this real gate is diagnostic.")
    print("It does NOT claim Z0 convergence until observed XYZ truth is validated.")
    print("ROLLING_3POINT_Z0_DIAGNOSTIC=COMPLETE")

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

    latest_pred = None
    for frame_no in range(a.start_frame, a.end_frame + 1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None:
            hit = rec["hit"]
            if hit.x is not None and hit.y is not None:
                cv2.circle(img, (round(hit.x), round(hit.y)), 10, (255,255,255), 2, cv2.LINE_AA)
            if "prediction" in rec:
                latest_pred = rec["prediction"]

        cv2.rectangle(img, (8,8), (630,88), (0,0,0), -1)
        cv2.putText(img, f"frame {frame_no}  RC6 ROLLING 3-POINT Z0", (18,31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        if latest_pred is None:
            text = "collecting 3 post-change points..."
        else:
            text = (
                f"Z0=({latest_pred.landing.x:.1f}, {latest_pred.landing.y:.1f}) BU  "
                f"dt={latest_pred.time_to_z0_s:.3f}s"
            )
            lx, ly = floor_to_image(latest_pred.landing, image_points, court)
            if -50 <= lx < width + 50 and -50 <= ly < height + 50:
                cv2.circle(img, (round(lx), round(ly)), 8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (18,58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, "ENGINEERING ONLY - predicted floor point", (18,79), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    cap.release()

    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("ROLLING_3POINT_Z0_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
