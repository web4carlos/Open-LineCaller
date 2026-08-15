from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np

BALL_DIAMETER_M = 0.074  # visual/physical target for one DCF impact cell

def load_frame(video, frame):
    cap=cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Could not read frame {frame}")
    return img

def load_points(path):
    d=json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(d["image_points"], dtype=np.float32)

def edge_point(pts, side, t):
    nl,nr,fr,fl=pts
    a,b={"left":(nl,fl),"right":(nr,fr),"near":(nl,nr),"far":(fl,fr)}[side]
    return a*(1-t)+b*t

def outward_unit(pts, p):
    center=pts.mean(axis=0)
    v=p-center
    n=float(np.linalg.norm(v))
    return v/(n if n else 1.0)

def local_ball_px(pts, side, t):
    # Estimate apparent ball diameter from local court perspective.
    # Court width is 20 ft = 6.096 m. Use visible local transverse width.
    nl,nr,fr,fl=pts
    left=nl*(1-t)+fl*t
    right=nr*(1-t)+fr*t
    court_width_px=float(np.linalg.norm(right-left))
    return court_width_px*(BALL_DIAMETER_M/6.096)

def glow_tile(img, center, diameter_px):
    cx,cy=map(float,center)
    r=max(2.0, diameter_px/2.0)

    # Floor footprint: compact perspective ellipse, one ball-sized cell.
    axes=(max(2,int(r)), max(2,int(r*0.52)))

    glow=np.zeros_like(img)
    for scale,alpha in ((3.0,.10),(2.2,.16),(1.55,.25)):
        layer=np.zeros_like(img)
        ax=(max(2,int(axes[0]*scale)),max(2,int(axes[1]*scale)))
        cv2.ellipse(layer,(int(cx),int(cy)),ax,0,0,360,(0,0,255),-1,cv2.LINE_AA)
        glow=cv2.addWeighted(glow,1.0,layer,alpha,0)

    out=cv2.addWeighted(img,1.0,glow,.9,0)
    core=np.zeros_like(img)
    cv2.ellipse(core,(int(cx),int(cy)),axes,0,0,360,(0,0,255),-1,cv2.LINE_AA)
    out=cv2.addWeighted(out,1.0,core,.72,0)
    cv2.ellipse(out,(int(cx),int(cy)),axes,0,0,360,(245,245,255),1,cv2.LINE_AA)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True)
    ap.add_argument("--calibration",required=True)
    ap.add_argument("--frame",type=int,required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--side",choices=["left","right","near","far"],default="right")
    ap.add_argument("--edge-position",type=float,default=.58)
    ap.add_argument("--outside-ball-widths",type=float,default=1.35)
    a=ap.parse_args()

    img=load_frame(a.video,a.frame)
    pts=load_points(a.calibration)
    t=max(0.0,min(1.0,a.edge_position))
    edge=edge_point(pts,a.side,t)
    ball_px=local_ball_px(pts,a.side,t)
    contact=edge+outward_unit(pts,edge)*(ball_px*a.outside_ball_widths)

    outimg=glow_tile(img,contact,ball_px)

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fn=out/"dcf_clean_out_impact.jpg"
    cv2.imwrite(str(fn),outimg)

    print("DCF CLEAN IMPACT")
    print("dcf_visible=False")
    print("grid_visible=False")
    print("wave_visible=False")
    print("hud_visible=False")
    print("contact_plane=Z=0")
    print("region=EXTERNAL")
    print("impact_cells_visible=1")
    print(f"impact_cell_diameter_m={BALL_DIAMETER_M:.3f}")
    print(f"impact_cell_apparent_px={ball_px:.2f}")
    print("impact_glow=RED_DIFFUSED")
    print("audio_event=OUT")
    print(f"image={fn}")

if __name__=="__main__":
    main()

