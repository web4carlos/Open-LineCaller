from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from linecaller.calibration_center import AutoCourtCalibrator
from linecaller.court_geometry import CourtCalibration


POINT_NAMES = [
    "NEAR-LEFT",
    "NEAR-RIGHT",
    "FAR-RIGHT",
    "FAR-LEFT",
]


def read_frame(video, frame_no):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Cannot read frame {frame_no}")

    return frame


def draw_quad(frame, calibration, color=(255,255,0)):
    out = frame.copy()
    pts = [
        (int(round(x)), int(round(y)))
        for x,y in calibration.image_points
    ]

    for i in range(4):
        cv2.line(
            out,
            pts[i],
            pts[(i+1)%4],
            color,
            3,
            cv2.LINE_AA,
        )

    for i,(x,y) in enumerate(pts):
        cv2.circle(out,(x,y),8,(0,255,255),-1)
        cv2.putText(
            out,
            str(i+1),
            (x+10,y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2,
            cv2.LINE_AA,
        )

    return out


def manual_mode(frame, output_path):
    points = []
    base = frame.copy()
    display = base.copy()
    window = "Manual Court Calibration"

    def redraw():
        nonlocal display
        display = base.copy()

        for i,(x,y) in enumerate(points):
            cv2.circle(display,(int(x),int(y)),8,(0,255,255),-1)
            cv2.putText(
                display,
                str(i+1),
                (int(x)+10,int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2,
                cv2.LINE_AA,
            )

        next_name = (
            POINT_NAMES[len(points)]
            if len(points)<4
            else "DONE - press S"
        )

        cv2.rectangle(display,(10,10),(900,92),(0,0,0),-1)
        cv2.putText(
            display,
            (
                f"Click {len(points)+1}/4: {next_name}"
                if len(points)<4 else next_name
            ),
            (25,45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "U undo | R reset | S save | Q quit",
            (25,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220,220,220),
            1,
            cv2.LINE_AA,
        )

    def on_mouse(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points)<4:
            points.append((float(x),float(y)))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window,on_mouse)
    redraw()

    while True:
        cv2.imshow(window,display)
        key=cv2.waitKey(20)&0xFF

        if key==ord("q"):
            break
        if key==ord("u") and points:
            points.pop(); redraw()
        if key==ord("r"):
            points.clear(); redraw()
        if key==ord("s"):
            if len(points)!=4:
                print("Need exactly 4 points.")
                continue
            c=CourtCalibration(image_points=tuple(points))
            c.save(output_path)
            print(f"saved={output_path}")
            break

    cv2.destroyWindow(window)


def auto_mode(frame, output_path, min_confidence):
    calibrator = AutoCourtCalibrator(
        min_confidence=min_confidence
    )
    result = calibrator.detect(frame)

    print(
        f"auto_success={result.success} "
        f"confidence={result.confidence:.3f} "
        f"reason={result.reason}"
    )

    if result.calibration is None:
        return False

    preview = draw_quad(
        frame,
        result.calibration,
        (255,255,0),
    )

    cv2.rectangle(preview,(10,10),(930,95),(0,0,0),-1)
    cv2.putText(
        preview,
        f"AUTO CALIBRATION confidence={result.confidence*100:.1f}%",
        (25,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255,255,255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "A accept | R retry | M manual | Q quit",
        (25,78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220,220,220),
        1,
        cv2.LINE_AA,
    )

    window="Auto Court Calibration Preview"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(window,preview)
        key=cv2.waitKey(20)&0xFF

        # Accept using A/a, ENTER or SPACE.
        if key in (ord("a"), ord("A"), 13, 32):
            print(f"accept_key={key}", flush=True)
            result.calibration.save(output_path)
            print(f"saved={output_path}", flush=True)
            cv2.destroyWindow(window)
            return True

        if key==ord("m"):
            cv2.destroyWindow(window)
            return False

        if key==ord("r"):
            cv2.destroyWindow(window)
            return None

        if key==ord("q"):
            cv2.destroyWindow(window)
            raise SystemExit(0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--frame",type=int,default=100)
    p.add_argument("--min-confidence",type=float,default=0.55)
    a=p.parse_args()

    Path(a.output).parent.mkdir(parents=True,exist_ok=True)

    frame_no=a.frame

    while True:
        frame=read_frame(a.video,frame_no)

        print("")
        print("OPEN-LINECALLER CALIBRATION CENTER")
        print("1 = AUTO CALIBRATION")
        print("2 = MANUAL CALIBRATION")
        print("")

        # Product default: AUTO first.
        choice=input("Choose [1]: ").strip() or "1"

        if choice=="2":
            manual_mode(frame,a.output)
            return

        result=auto_mode(
            frame,
            a.output,
            a.min_confidence,
        )

        if result is True:
            return

        if result is False:
            print("Switching to MANUAL CALIBRATION.")
            manual_mode(frame,a.output)
            return

        # Retry: move forward 30 frames to seek a cleaner view.
        frame_no += 30
        print(f"Retrying auto calibration at frame {frame_no}...")


if __name__=="__main__":
    main()

