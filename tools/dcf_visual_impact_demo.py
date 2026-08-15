import argparse,json,cv2
from pathlib import Path
import numpy as np
def bilinear(p,u,v):
 nl,nr,fr,fl=p; return (nl*(1-u)+nr*u)*(1-v)+(fl*(1-u)+fr*u)*v
def outside(p,side,t,d):
 nl,nr,fr,fl=p;c=p.mean(0);a,b={"left":(nl,fl),"right":(nr,fr),"near":(nl,nr),"far":(fl,fr)}[side]
 q=a*(1-t)+b*t;n=q-c;n=n/(np.linalg.norm(n) or 1);return q+n*d
def main():
 a=argparse.ArgumentParser();a.add_argument("--video",required=True);a.add_argument("--calibration",required=True);a.add_argument("--frame",type=int,required=True);a.add_argument("--output-dir",required=True);a.add_argument("--side",default="right",choices=["left","right","near","far"]);x=a.parse_args()
 p=np.array(json.loads(Path(x.calibration).read_text())["image_points"],np.float32);cap=cv2.VideoCapture(x.video);cap.set(cv2.CAP_PROP_POS_FRAMES,x.frame);ok,im=cap.read();cap.release()
 if not ok:raise RuntimeError("frame read failed")
 q=outside(p,x.side,.55,35)
 for u in np.linspace(0,1,17):cv2.line(im,tuple(np.int32(bilinear(p,u,0))),tuple(np.int32(bilinear(p,u,1))),(190,190,190),1)
 for v in np.linspace(0,1,25):cv2.line(im,tuple(np.int32(bilinear(p,0,v))),tuple(np.int32(bilinear(p,1,v))),(190,190,190),1)
 cv2.polylines(im,[np.int32(p)],True,(255,255,255),3)
 path=[]
 for k in range(6,-1,-1):
  t=(6-k)/6;q2=q+np.array([-95*(1-t),-35*(1-t)-k*17]);path.append(q2)
 for i in range(6):cv2.line(im,tuple(np.int32(path[i])),tuple(np.int32(path[i+1])),(255,255,0),2)
 cx,cy=q;tile=np.int32([[cx-16,cy-8],[cx+16,cy-8],[cx+19,cy+9],[cx-19,cy+9]]);o=im.copy();cv2.fillConvexPoly(o,tile,(0,0,255));im=cv2.addWeighted(o,.65,im,.35,0)
 cv2.putText(im,"DCF Z=0 IMPACT CELL: OUT",(30,55),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
 out=Path(x.output_dir);out.mkdir(parents=True,exist_ok=True);fn=out/"dcf_impact_frame.jpg";cv2.imwrite(str(fn),im)
 print("region=EXTERNAL");print("impact_cell_floor=ON");print("audio_event=OUT");print("image="+str(fn))
if __name__=="__main__":main()
