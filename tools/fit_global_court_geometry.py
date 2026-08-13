from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from linecaller.calibration_center.global_geometry import GlobalPickleballGeometryFitter

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--frame",type=int,default=100)
    p.add_argument("--search-px",type=int,default=18)
    p.add_argument("--step-px",type=int,default=6)
    p.add_argument("--min-gain",type=float,default=.025)
    a=p.parse_args()

    data=json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    cap=cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit("Could not read requested frame.")

    fitter=GlobalPickleballGeometryFitter(
        search_px=a.search_px, step_px=a.step_px, min_gain=a.min_gain
    )
    r=fitter.fit(frame,data["image_points"])

    print(f"accepted={r.accepted}")
    print(f"original_score={r.original_score:.3f}")
    print(f"best_score={r.best_score:.3f}")
    print(f"gain={r.gain:.3f}")
    print(f"candidate_count={r.candidate_count}")
    print(f"reason={r.reason}")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    new=dict(data)
    new["image_points"]=[list(p) for p in r.image_points]
    new["global_geometry"]={
        "accepted":r.accepted,
        "original_score":r.original_score,
        "best_score":r.best_score,
        "gain":r.gain,
        "candidate_count":r.candidate_count,
        "reason":r.reason,
    }
    out.write_text(json.dumps(new,indent=2),encoding="utf-8")
    print(f"output={out}")

if __name__=="__main__":
    main()
