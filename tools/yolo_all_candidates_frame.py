from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--video",required=True)
    p.add_argument("--frame",type=int,required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--conf",type=float,default=0.05)
    a=p.parse_args()

    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Cannot read frame {a.frame}")

    model=YOLO(a.model)
    result=model.predict(
        frame,
        conf=a.conf,
        imgsz=a.imgsz,
        device="cpu",
        verbose=False,
    )[0]

    rows=[]

    if result.boxes is not None:
        for i in range(len(result.boxes)):
            conf=float(result.boxes.conf[i].item())
            cls=int(result.boxes.cls[i].item()) if result.boxes.cls is not None else -1
            x1,y1,x2,y2=map(float,result.boxes.xyxy[i].cpu().numpy().tolist())
            x=(x1+x2)/2.0
            y=(y1+y2)/2.0
            w=x2-x1
            h=y2-y1

            rows.append({
                "frame":a.frame,
                "candidate_id":i,
                "class_id":cls,
                "confidence":round(conf,6),
                "x":round(x,3),
                "y":round(y,3),
                "width":round(w,3),
                "height":round(h,3),
                "x1":round(x1,3),
                "y1":round(y1,3),
                "x2":round(x2,3),
                "y2":round(y2,3),
            })

            cv2.rectangle(
                frame,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,255,255),
                2,
            )
            cv2.putText(
                frame,
                f"#{i} {conf:.2f}",
                (int(x1),max(20,int(y1)-7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .6,
                (0,255,255),
                2,
                cv2.LINE_AA,
            )

    csv_path=out/f"frame_{a.frame}_candidates.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        fields=[
            "frame","candidate_id","class_id","confidence",
            "x","y","width","height","x1","y1","x2","y2"
        ]
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    image_path=out/f"frame_{a.frame}_all_candidates.jpg"
    cv2.imwrite(str(image_path),frame)

    print(f"frame={a.frame}")
    print(f"candidates={len(rows)}")
    print(f"csv={csv_path}")
    print(f"image={image_path}")

    for r in rows:
        print(
            f"id={r['candidate_id']} "
            f"conf={r['confidence']} "
            f"xy=({r['x']},{r['y']}) "
            f"size=({r['width']},{r['height']})"
        )

if __name__=="__main__":
    main()
