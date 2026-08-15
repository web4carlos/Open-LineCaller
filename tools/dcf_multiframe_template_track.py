from __future__ import annotations

import argparse
import csv
from pathlib import Path
import cv2

from linecaller.dcf.multiframe_template_tracker import DCFMultiFrameTemplateTracker


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--template",required=True)
    p.add_argument("--start-frame",type=int,required=True)
    p.add_argument("--frame-count",type=int,default=60)
    p.add_argument("--initial-x",type=float,required=True)
    p.add_argument("--initial-y",type=float,required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--min-score",type=float,default=.36)
    a=p.parse_args()

    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    tracker=DCFMultiFrameTemplateTracker(min_score=a.min_score)
    points=tracker.track(
        a.video,
        a.template,
        start_frame=a.start_frame,
        frame_count=a.frame_count,
        initial_center=(a.initial_x,a.initial_y),
    )

    csv_path=out/"dcf_template_track.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=["frame","found","x","y","score","scale","roi"])
        wr.writeheader()
        for pnt in points:
            wr.writerow({
                "frame":pnt.frame,
                "found":pnt.found,
                "x":"" if pnt.x is None else round(pnt.x,3),
                "y":"" if pnt.y is None else round(pnt.y,3),
                "score":round(pnt.score,6),
                "scale":"" if pnt.scale is None else pnt.scale,
                "roi":pnt.roi,
            })

    # Render a compact overlay video for visual verification.
    cap=cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.start_frame)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_path=out/"dcf_template_track_overlay.mp4"
    writer=cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height),
    )

    for pnt in points:
        ok,frame=cap.read()
        if not ok:
            break
        x1,y1,x2,y2=pnt.roi
        cv2.rectangle(frame,(x1,y1),(x2,y2),(100,100,100),1)

        if pnt.found:
            cv2.circle(frame,(round(pnt.x),round(pnt.y)),10,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(
                frame,
                f"MATCH {pnt.score:.2f} scale={pnt.scale}",
                (max(10,round(pnt.x)-100),max(30,round(pnt.y)-18)),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2,cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                "NO MATCH - DCF ROI EXPANDING",
                (25,45),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,165,255),2,cv2.LINE_AA
            )

        writer.write(frame)

    cap.release()
    writer.release()

    found=sum(1 for p in points if p.found)
    print("DCF MULTIFRAME TEMPLATE TRACK")
    print(f"frames={len(points)}")
    print(f"found={found}")
    print(f"coverage={(found/len(points)) if points else 0:.3%}")
    print(f"csv={csv_path}")
    print(f"overlay={overlay_path}")

    for pnt in points:
        print(
            f"frame={pnt.frame} found={pnt.found} "
            f"score={pnt.score:.4f} scale={pnt.scale} "
            f"x={pnt.x} y={pnt.y}"
        )


if __name__=="__main__":
    main()
