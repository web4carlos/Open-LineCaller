import argparse,cv2
from pathlib import Path
from linecaller.dcf.template_matcher import DCFBallTemplateMatcher
p=argparse.ArgumentParser();p.add_argument("--video",required=True);p.add_argument("--frame",type=int,required=True);p.add_argument("--template",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--roi",nargs=4,type=int,required=True);p.add_argument("--min-score",type=float,default=.42);a=p.parse_args()
cap=cv2.VideoCapture(a.video);cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,img=cap.read();cap.release()
if not ok:raise RuntimeError("frame read failed")
t=cv2.imread(a.template);r=DCFBallTemplateMatcher(a.min_score).match(img,t,a.roi);vis=img.copy();x1,y1,x2,y2=r.roi;cv2.rectangle(vis,(x1,y1),(x2,y2),(160,160,160),1)
if r.found:cv2.circle(vis,(round(r.x),round(r.y)),10,(0,0,255),2)
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);fn=out/f"frame_{a.frame}_dcf_template_match.jpg";cv2.imwrite(str(fn),vis)
print(f"found={r.found}\nscore={r.score:.6f}\nscale={r.scale}\nx={r.x}\ny={r.y}\nimage={fn}")
