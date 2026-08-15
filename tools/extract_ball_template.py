import argparse,cv2
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--video",required=True);p.add_argument("--frame",type=int,required=True);p.add_argument("--x",type=int,required=True);p.add_argument("--y",type=int,required=True);p.add_argument("--size",type=int,default=32);p.add_argument("--output",required=True);a=p.parse_args()
cap=cv2.VideoCapture(a.video);cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,img=cap.read();cap.release()
if not ok:raise RuntimeError("frame read failed")
r=a.size//2;crop=img[max(0,a.y-r):a.y+r,max(0,a.x-r):a.x+r];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out),crop);print("template="+str(out))
