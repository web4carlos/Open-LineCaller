from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

def load_points(path):
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["image_points"],np.float32)

def floor_point(p,u,v):
    nl,nr,fr,fl=p
    return ((nl*(1-u)+nr*u)*(1-v)+(fl*(1-u)+fr*u)*v)

def outside_floor(p,side,t,d):
    nl,nr,fr,fl=p; c=p.mean(0)
    a,b={"left":(nl,fl),"right":(nr,fr),"near":(nl,nr),"far":(fl,fr)}[side]
    q=a*(1-t)+b*t; n=q-c; n=n/(np.linalg.norm(n) or 1.0)
    return q+n*d

def height_px(v,z):
    # Perspective-aware visual height: strongest near camera, approaches zero far away.
    return z*(34.0*(1-v)+5.0*v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True)
    ap.add_argument("--calibration",required=True)
    ap.add_argument("--frame",type=int,required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--side",choices=["left","right","near","far"],default="right")
    a=ap.parse_args()

    p=load_points(a.calibration)
    cap=cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError("Could not read requested frame")

    # Faint full field: floor mesh only as reference, deliberately subtle.
    for u in np.linspace(0,1,13):
        cv2.line(img,tuple(np.int32(floor_point(p,u,0))),tuple(np.int32(floor_point(p,u,1))),(90,90,90),1,cv2.LINE_AA)
    for v in np.linspace(0,1,19):
        cv2.line(img,tuple(np.int32(floor_point(p,0,v))),tuple(np.int32(floor_point(p,1,v))),(90,90,90),1,cv2.LINE_AA)
    cv2.polylines(img,[np.int32(p)],True,(220,220,220),2,cv2.LINE_AA)

    # Simulated ball wave moving toward external right field and descending.
    # Each state illuminates only a local 3-D neighborhood.
    states=[
        (.72,.52,5.0),(.78,.54,4.2),(.84,.56,3.4),
        (.91,.58,2.5),(.98,.60,1.4)
    ]
    active=[]
    for si,(u,v,z) in enumerate(states):
        alpha=(si+1)/len(states)
        for du in (-.025,0,.025):
            for dv in (-.018,0,.018):
                for dz in (-.55,0,.55):
                    uu=u+du; vv=v+dv; zz=max(.15,z+dz)
                    base=floor_point(p,uu,vv)
                    top=base+np.array([0,-height_px(vv,zz)],np.float32)
                    active.append((top,alpha))
    for q,alpha in active:
        r=max(2,int(3+3*alpha))
        # Bright active cells; full volume itself remains implicit/faint.
        cv2.rectangle(img,(int(q[0]-r),int(q[1]-r)),(int(q[0]+r),int(q[1]+r)),(0,220,255),1,cv2.LINE_AA)

    # Path through centers of illuminated wave.
    centers=[]
    for u,v,z in states:
        b=floor_point(p,u,v); centers.append(b+np.array([0,-height_px(v,z)],np.float32))
    for x,y in zip(centers,centers[1:]):
        cv2.line(img,tuple(np.int32(x)),tuple(np.int32(y)),(255,255,0),2,cv2.LINE_AA)

    # External z=0 impact cell.
    contact=outside_floor(p,a.side,.58,38)
    cv2.line(img,tuple(np.int32(centers[-1])),tuple(np.int32(contact)),(255,255,0),2,cv2.LINE_AA)
    cx,cy=contact
    tile=np.int32([[cx-17,cy-8],[cx+17,cy-8],[cx+20,cy+10],[cx-20,cy+10]])
    ov=img.copy(); cv2.fillConvexPoly(ov,tile,(0,0,255),cv2.LINE_AA)
    img=cv2.addWeighted(ov,.72,img,.28,0)
    cv2.polylines(img,[tile],True,(255,255,255),2,cv2.LINE_AA)

    cv2.putText(img,"DCF DYNAMIC VOLUME",(28,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,"ACTIVE WAVE -> Z=0 -> IMPACT CELL -> OUT",(28,80),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,220,255),2,cv2.LINE_AA)

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fn=out/"dcf_dynamic_volume.jpg"; cv2.imwrite(str(fn),img)
    print("DCF DYNAMIC VOLUME POC")
    print(f"active_cells={len(active)}")
    print("full_field=IMPLICIT_FAINT")
    print("players_modeled=False")
    print("contact_plane=Z=0")
    print("impact_region=EXTERNAL")
    print("impact_cell_floor=ON")
    print("audio_event=OUT")
    print(f"image={fn}")

if __name__=="__main__": main()
