from __future__ import annotations
import argparse, cv2, numpy as np
from pathlib import Path
from linecaller.court_geometry import CourtCalibration

def court_to_image(calibration, points):
    src=np.array([[0.,44.],[20.,44.],[20.,0.],[0.,0.]],dtype=np.float32)
    dst=np.array(calibration.image_points,dtype=np.float32)
    H=cv2.getPerspectiveTransform(src,dst)
    pts=np.array([[[float(x),float(y)]] for x,y in points],dtype=np.float32)
    tr=cv2.perspectiveTransform(pts,H)
    return [(int(round(p[0][0])),int(round(p[0][1]))) for p in tr]

def draw_segment(frame,calibration,p1,p2,label=None):
    a,b=court_to_image(calibration,[p1,p2])
    cv2.line(frame,a,b,(255,255,0),3,cv2.LINE_AA)
    if label:
        mx=(a[0]+b[0])//2; my=(a[1]+b[1])//2
        cv2.putText(frame,label,(mx+6,my-6),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,255,255),2,cv2.LINE_AA)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--frame",type=int,default=100)
    p.add_argument("--output",required=True)
    a=p.parse_args()

    calibration=CourtCalibration.load(a.calibration)
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open video: {a.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit(f"Cannot read frame {a.frame}")

    draw_segment(frame,calibration,(0,0),(20,0),"FAR BASELINE")
    draw_segment(frame,calibration,(0,44),(20,44),"NEAR BASELINE")
    draw_segment(frame,calibration,(0,0),(0,44),"LEFT SIDELINE")
    draw_segment(frame,calibration,(20,0),(20,44),"RIGHT SIDELINE")
    draw_segment(frame,calibration,(0,15),(20,15),"FAR NVZ")
    draw_segment(frame,calibration,(0,22),(20,22),"NET")
    draw_segment(frame,calibration,(0,29),(20,29),"NEAR NVZ")
    draw_segment(frame,calibration,(10,0),(10,15),"FAR CENTER")
    draw_segment(frame,calibration,(10,29),(10,44),"NEAR CENTER")

    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(a.output,frame)
    print(f"output={a.output}")

if __name__=="__main__":
    main()
