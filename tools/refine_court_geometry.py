from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from linecaller.calibration_center.refinement import CourtGeometryRefiner

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--frame",type=int,default=100)
    p.add_argument("--search-radius",type=int,default=24)
    p.add_argument("--min-gain",type=float,default=.025)
    a=p.parse_args()

    data=json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    points=data["image_points"]

    cap=cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit("Could not read requested video frame.")

    refiner=CourtGeometryRefiner(a.search_radius,a.min_gain)
    r=refiner.refine(frame,points)

    print(f"accepted={r.accepted}")
    print(f"original_score={r.original_score:.3f}")
    print(f"refined_score={r.refined_score:.3f}")
    print(f"gain={r.confidence_gain:.3f}")
    print(f"reason={r.reason}")

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)

    new_data=dict(data)
    new_data["image_points"]=[list(p) for p in r.image_points]
    new_data["geometry_refinement"]={
        "accepted":r.accepted,
        "original_score":r.original_score,
        "refined_score":r.refined_score,
        "gain":r.confidence_gain,
        "reason":r.reason,
    }
    out.write_text(json.dumps(new_data,indent=2),encoding="utf-8")
    print(f"output={out}")

if __name__=="__main__":
    main()
