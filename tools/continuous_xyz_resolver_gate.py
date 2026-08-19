from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from linecaller.dcf.ball_color_residual_isolator import (
    BallColorResidualIsolator,
    LockedResidualColorProfile,
)
from linecaller.dcf.ball_in_mesh_search import BallInMeshSearcher
from linecaller.dcf.models import CellIndex, DCFConfig
from linecaller.dcf.residual_ball_projection import ResidualBallProjectionEstimator
from linecaller.flight_paths.continuous_xyz import (
    ContinuousXYZResolver,
    CourtProjectionCalibration,
    ProjectionMeasurement,
)
from linecaller.flight_paths.models import CourtFrame, Point3D, TimedPoint3D
from linecaller.flight_paths.rolling_z0 import RollingZ0Tracker


# Same-ball sequence visually confirmed by the user in RC5/RC7.2.
RC5_CONFIRMED = (
    (3813, CellIndex(69, 60, 6), 336.1, 251.8, 0.760, "TRANSITION_LOCAL"),
    (3814, CellIndex(70, 65, 9), 340.8, 225.8, 0.758, "TRANSITION_LOCAL"),
    (3815, CellIndex(71, 70, 12), 344.3, 200.5, 0.477, "NEW_PATH_CONFIRMED"),
    (3816, CellIndex(71, 70, 15), 345.3, 181.0, 0.644, "NEW_PATH_CONFIRMED"),
    (3817, CellIndex(71, 65, 18), 347.5, 166.2, 0.750, "NEW_PATH_CONFIRMED"),
    (3818, CellIndex(71, 61, 20), 349.2, 156.5, 0.659, "NEW_PATH_CONFIRMED"),
    (3819, CellIndex(73, 61, 19), 362.6, 163.2, 0.488, "NEW_PATH_CONFIRMED"),
    (3820, CellIndex(72, 66, 22), 352.6, 138.9, 0.612, "NEW_PATH_CONFIRMED"),
    (3821, CellIndex(72, 68, 23), 352.7, 130.7, 0.577, "NEW_PATH_CONFIRMED"),
    (3822, CellIndex(72, 66, 24), 353.6, 125.7, 0.506, "NEW_PATH_CONFIRMED"),
)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_frame(cap: cv2.VideoCapture, frame_no: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_no}")
    return frame


def crop_rgb_with_origin(frame_bgr, cx: float, cy: float, half: int):
    h, w = frame_bgr.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    x1, y1 = max(0, ix-half), max(0, iy-half)
    x2, y2 = min(w, ix+half+1), min(h, iy+half+1)
    rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return rgb, (float(x1), float(y1))


def cell_to_anchor(cell: CellIndex) -> Point3D:
    cfg = DCFConfig()
    court = CourtFrame()
    sx = court.width_bu / cfg.court_x_cells
    sy = court.length_bu / cfg.court_y_cells
    return Point3D(
        (cell.x - cfg.margin_cells + 0.5) * sx,
        (cell.y - cfg.margin_cells + 0.5) * sy,
        float(cell.z),
    )


def draw_cross(img, pt, radius=7, thickness=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x-radius,y), (x+radius,y), (255,255,255), thickness, cv2.LINE_AA)
    cv2.line(img, (x,y-radius), (x,y+radius), (255,255,255), thickness, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--background-frame", type=int, default=3780)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    data = load_json(a.calibration)
    calibration = CourtProjectionCalibration.from_json_data(data)

    template_rgb = np.asarray(Image.open(a.template).convert("RGB"), dtype=np.uint8)
    profile = LockedResidualColorProfile.from_template(template_rgb)
    isolator = BallColorResidualIsolator(profile)
    estimator = ResidualBallProjectionEstimator()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Invalid FPS")

    background = read_frame(cap, a.background_frame)
    legacy_projector = BallInMeshSearcher(np.asarray(data["image_points"], dtype=np.float64)).projector

    records=[]
    scale_window=deque(maxlen=3)
    resolver=ContinuousXYZResolver(calibration, measurement_weight=0.72)
    rolling=RollingZ0Tracker()
    last_z0=None

    for i,(frame_no,cell,px,py,id_score,state) in enumerate(RC5_CONFIRMED):
        frame=read_frame(cap,frame_no)
        expected_d=float(legacy_projector.project(cell).expected_diameter_px)
        half=max(10,int(round(expected_d*2.0)))
        current_rgb,origin=crop_rgb_with_origin(frame,px,py,half)
        background_rgb,bg_origin=crop_rgb_with_origin(background,px,py,half)
        if origin != bg_origin:
            raise RuntimeError("ROI alignment mismatch")
        ev=isolator.isolate(current_rgb,background_rgb,expected_diameter_px=expected_d)
        proj=estimator.measure(ev,roi_origin_xy=origin)
        if not proj.valid or proj.center_xy_global is None:
            raise RuntimeError(f"Invalid RC7.2 projection at frame {frame_no}")
        depth_scale=proj.moment_minor_diameter_px
        if depth_scale is None or not math.isfinite(depth_scale) or depth_scale <= 0:
            raise RuntimeError(f"Invalid projection scale at frame {frame_no}")
        scale_window.append(float(depth_scale))
        smoothed_scale=float(np.median(np.asarray(scale_window,dtype=float)))
        t=frame_no/fps
        m=ProjectionMeasurement(proj.center_xy_global,smoothed_scale,t)
        if i==0:
            anchor=TimedPoint3D(cell_to_anchor(cell),t)
            resolved=resolver.prime(anchor,m)
        else:
            resolved=resolver.resolve(m)
        z0=rolling.add(resolved.resolved)
        shift=None
        if z0 is not None and last_z0 is not None:
            shift=math.hypot(z0.landing.x-last_z0.x,z0.landing.y-last_z0.y)
        if z0 is not None:
            last_z0=z0.landing
        records.append({
            "frame":frame_no,"cell":cell,"px":px,"py":py,"id":id_score,"state":state,
            "projection":proj,"depth_scale":depth_scale,"smooth_scale":smoothed_scale,
            "resolution":resolved,"z0":z0,"z0_shift":shift,
        })
    cap.release()

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC8 - CONTINUOUS XYZ RESOLVER")
    print("============================================================")
    print()
    print(f"FPS.............................. {fps:.6f}")
    print(f"CALIBRATION MODE................. {calibration.mode.value}")
    print("FULL/HALF ENGINE................. SAME XYZ CONTRACT")
    print("BALL IDENTITY.................... WHO ONLY")
    print("RESIDUAL CENTER.................. RC7.2 (u,v)")
    print("DEPTH EVIDENCE................... MOMENT-MINOR SCALE / 3-FRAME MEDIAN")
    print("TIME............................. VIDEO FPS")
    print("INITIAL XYZ...................... ONE TRUSTED SAME-BALL ANCHOR")
    print("MESH/CELLS DEFINE PATH........... NO")
    print("COMPLETE PATH.................... NOT CALCULATED")
    print("ROLLING Z0....................... RC6 DOWNSTREAM DIAGNOSTIC")
    print()
    print(f"RUNTIME SCALE GAIN............... {resolver.scale_gain:.5f}")
    print()
    print("FRAME | CENTER(u,v)       | SCALE | RAW XYZ                  | RESOLVED XYZ             | REPROJ | Z0")
    print("-"*145)
    shifts=[]
    for r in records:
        q=r["projection"]
        e=r["resolution"]
        gx,gy=q.center_xy_global
        raw=e.raw_projection_xyz
        p=e.resolved.point
        z0=r["z0"]
        if z0 is None:
            ztxt="warming"
        else:
            ztxt=f"({z0.landing.x:6.2f},{z0.landing.y:7.2f},0)"
            if r["z0_shift"] is not None:
                shifts.append(r["z0_shift"])
                ztxt += f" shift={r['z0_shift']:6.2f}"
        print(
            f"{r['frame']:5d} | ({gx:7.2f},{gy:7.2f}) | {r['smooth_scale']:5.2f} | "
            f"({raw.x:6.2f},{raw.y:7.2f},{raw.z:6.2f}) | "
            f"({p.x:6.2f},{p.y:7.2f},{p.z:6.2f}) | "
            f"{e.center_reprojection_error_px:6.2f}px | {ztxt}"
        )
    print()
    print(f"VALID XYZ RESOLUTIONS............ {len(records)}/{len(RC5_CONFIRMED)}")
    print("XYZ IS CONTINUOUS................. YES / SUB-CELL")
    print("Z0 CONVERGENCE CLAIM.............. NO - inspect shifts first")
    if shifts:
        print(f"Z0 SHIFT FIRST/LAST............... {shifts[0]:.2f} / {shifts[-1]:.2f} BU")
        print(f"Z0 SHIFT MEDIAN................... {float(np.median(shifts)):.2f} BU")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    first,last=RC5_CONFIRMED[0][0],RC5_CONFIRMED[-1][0]
    by_frame={r['frame']:r for r in records}
    cap=cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w=560
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(fw+panel_w,fh))
    if not writer.isOpened(): raise RuntimeError("Could not open output video")
    last_rec=None
    for frame_no in range(first,last+1):
        ok,img=cap.read()
        if not ok: break
        rec=by_frame.get(frame_no)
        if rec is not None:
            last_rec=rec
            draw_cross(img,rec['projection'].center_xy_global,6,2)
        canvas=np.zeros((fh,fw+panel_w,3),dtype=np.uint8); canvas[:,:fw]=img
        panel=canvas[:,fw:]
        cv2.putText(panel,"RC8 CONTINUOUS XYZ",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(panel,f"calibration={calibration.mode.value}",(12,54),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        if last_rec:
            e=last_rec['resolution']; p=e.resolved.point; raw=e.raw_projection_xyz
            lines=[
                f"frame {last_rec['frame']}",
                f"center=({last_rec['projection'].center_xy_global[0]:.2f},{last_rec['projection'].center_xy_global[1]:.2f}) px",
                f"scale={last_rec['smooth_scale']:.2f}px",
                f"raw XYZ=({raw.x:.2f},{raw.y:.2f},{raw.z:.2f}) BU",
                f"resolved=({p.x:.2f},{p.y:.2f},{p.z:.2f}) BU",
                f"reproj={e.center_reprojection_error_px:.2f}px",
            ]
            if last_rec['z0'] is not None:
                z=last_rec['z0'].landing
                lines.append(f"rolling Z0=({z.x:.2f},{z.y:.2f},0)")
                if last_rec['z0_shift'] is not None:
                    lines.append(f"Z0 shift={last_rec['z0_shift']:.2f} BU")
            y=92
            for text in lines:
                cv2.putText(panel,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA); y+=28
        writer.write(canvas)
    writer.release(); cap.release()

    print()
    print(f"output={out}")
    print("PRODUCTION_OVERLAY=NO")
    print("INITIAL_ANCHOR_IS_ENGINEERING_ONLY=YES")
    print("RC8_CONTINUOUS_XYZ_GATE=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
