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
from linecaller.flight_paths.models import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.transition_continuity import DirectionTransitionContinuity
from linecaller.flight_paths.hit_anchored_z0 import HitAnchoredRollingZ0


def load_points(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["image_points"], dtype=np.float64)


def cell_to_point(cell, cfg: DCFConfig, court: CourtFrame) -> Point3D:
    # Engineering observation adapter only.  It does not define the rolling
    # algorithm.  RC8 continuous-XYZ is deliberately NOT used in this gate.
    sx = court.width_bu / cfg.court_x_cells
    sy = court.length_bu / cfg.court_y_cells
    return Point3D(
        (cell.x - cfg.margin_cells + 0.5) * sx,
        (cell.y - cfg.margin_cells + 0.5) * sy,
        float(cell.z),
    )


def floor_to_image(point: Point3D, image_points, court: CourtFrame):
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
    rolling = HitAnchoredRollingZ0(gravity_bu_s2=court.gravity_bu_s2)

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
    hit_anchor_frame = None
    global_reacquisition_count = 0
    last_trusted = acquisition
    last_trusted_frame = a.start_frame

    def obs_from_hit(frame_no, hit):
        return TimedPoint3D(cell_to_point(hit.index, cfg, court), frame_no / fps)

    records.append((a.start_frame, acquisition, "OLD_FLIGHT"))

    for frame_no in range(a.start_frame + 1, a.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        if transition is None:
            step = old_tracker.track(frame, ball)
            if step.hit.found and step.hit.index is not None:
                last_trusted = step.hit
                last_trusted_frame = frame_no
                records.append((frame_no, step.hit, "OLD_FLIGHT"))
                continue

            # The old continuity broke at the direction-change boundary.
            # Anchor the NEW rolling flight at the last trusted same-ball
            # observation.  No global reacquisition is performed.
            transition = DirectionTransitionContinuity(searcher)
            transition.start(last_trusted)
            transition_start_frame = frame_no
            hit_anchor_frame = last_trusted_frame
            rolling.start_epoch(obs_from_hit(last_trusted_frame, last_trusted))

        tstep = transition.track(frame, ball)
        if not tstep.hit.found or tstep.hit.index is None:
            continue

        pred = rolling.add(obs_from_hit(frame_no, tstep.hit))
        state = "POST_HIT_POINT"
        if pred is not None:
            state = "ROLLING_Z0_ACTIVE"
            predictions.append((frame_no, pred))
        records.append((frame_no, tstep.hit, state))

    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC8.1 - HIT-ANCHORED ROLLING 3-POINT Z0")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print("PADDLE HIT....................... STARTS NEW ROLLING EPOCH")
    print("FIRST WINDOW..................... HIT + P1 + P2")
    print("NEXT WINDOW...................... P1 + P2 + P3")
    print("WINDOW HISTORY................... MAX 3 POINTS")
    print("FULL PATH........................ NOT CALCULATED")
    print("RC8 CONTINUOUS XYZ............... BYPASSED IN THIS GATE")
    print("GLOBAL REACQUISITION............. FORBIDDEN")
    print()
    print(f"DIRECTION-CHANGE BOUNDARY........ {transition_start_frame}")
    print(f"HIT ANCHOR FRAME................. {hit_anchor_frame}")
    print(f"GLOBAL REACQUISITION COUNT....... {global_reacquisition_count}")
    print()

    print("ROLLING ESTIMATES:")
    print("FRAME | 3-POINT WINDOW     | CURRENT P                  | predicted Z0              | shift")
    print("-" * 112)
    prev = None
    for frame_no, pred in predictions:
        frames = [round(o.t_s * fps) for o in pred.observations]
        shift = None if prev is None else distance_xy(pred.landing, prev)
        shift_text = "FIRST" if shift is None else f"{shift:6.2f} BU"
        print(
            f"{frame_no:5d} | {frames[0]:4d}-{frames[1]:4d}-{frames[2]:4d} | "
            f"({pred.current.x:6.2f},{pred.current.y:6.2f},{pred.current.z:5.2f}) | "
            f"({pred.landing.x:7.2f},{pred.landing.y:7.2f},0) | {shift_text}"
        )
        prev = pred.landing

    print()
    print("IMPORTANT:")
    print("This gate validates the HIT -> 3 -> shift-by-1 sequencing.")
    print("It does NOT accept RC8's collapsed XYZ solution.")
    print("Observed P values are engineering evidence from same-ball continuity.")
    print("Final Z0 accuracy is NOT claimed by this sequencing gate.")
    print("HIT_ANCHORED_ROLLING_SEQUENCE_GATE=COMPLETE")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_frame = {fr: (hit, state) for fr, hit, state in records}
    pred_by_frame = {fr: pred for fr, pred in predictions}
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open output video")

    latest_pred = None
    for frame_no in range(a.start_frame, a.end_frame + 1):
        ok, img = cap.read()
        if not ok:
            break
        rec = by_frame.get(frame_no)
        if rec is not None:
            hit, state = rec
            if hit.x is not None and hit.y is not None:
                cv2.circle(img, (round(hit.x), round(hit.y)), 10, (255,255,255), 2, cv2.LINE_AA)
        if frame_no in pred_by_frame:
            latest_pred = pred_by_frame[frame_no]

        cv2.rectangle(img, (8,8), (632,100), (0,0,0), -1)
        cv2.putText(img, f"frame {frame_no}  HIT-ANCHORED ROLLING Z0", (18,31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        if hit_anchor_frame is None or frame_no < hit_anchor_frame:
            line = "old flight - waiting for direction-change boundary"
        elif frame_no == hit_anchor_frame:
            line = f"HIT ANCHOR H0 = frame {hit_anchor_frame}"
        elif latest_pred is None:
            line = "collecting P1 + P2 after HIT..."
        else:
            line = f"rolling Z0=({latest_pred.landing.x:.1f},{latest_pred.landing.y:.1f}) BU"
            lx, ly = floor_to_image(latest_pred.landing, image_points, court)
            if -50 <= lx < width+50 and -50 <= ly < height+50:
                cv2.circle(img, (round(lx), round(ly)), 8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (18,59), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, "H0,P1,P2 -> P1,P2,P3 -> ... ; NO FULL PATH", (18,84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    cap.release()
    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("RC8_1_HIT_ANCHORED_ROLLING_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
