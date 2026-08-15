import argparse,json,cv2,numpy as np
from pathlib import Path
from linecaller.dcf.one_ball_field import OneBallOneDCF,DCFCell
p=argparse.ArgumentParser();p.add_argument("--video",required=True);p.add_argument("--frame",type=int,required=True);p.add_argument("--template",required=True);p.add_argument("--calibration",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--cell-px",type=int,default=48);p.add_argument("--min-score",type=float,default=.55);p.add_argument("--z0",action="store_true");a=p.parse_args()
cap=cv2.VideoCapture(a.video);cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,img=cap.read();cap.release()
if not ok:raise RuntimeError("frame read failed")
ball=cv2.imread(a.template);d=json.loads(Path(a.calibration).read_text());q=np.asarray(d["image_points"],np.float32);h,w=img.shape[:2];far=min(q[:,1]);near=max(q[:,1]);cells=[]
for ix,x1 in enumerate(range(0,w,a.cell_px)):
 for iy,y1 in enumerate(range(0,h,a.cell_px)):
  x2=min(w,x1+a.cell_px);y2=min(h,y1+a.cell_px);cx=(x1+x2)/2;cy=(y1+y2)/2
  inside=cv2.pointPolygonTest(q.reshape((-1,1,2)),(cx,cy),False)>=0
  depth=max(0,min(1,(cy-far)/max(1,near-far)));scale=.5+1.5*depth
  cells.append(DCFCell(ix,iy,x1,y1,x2,y2,inside,scale))
hit=OneBallOneDCF(a.min_score).search(img,ball,cells); final_out=bool(hit.found and hit.cell and not hit.cell.inside and a.z0)
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);fn=out/f"frame_{a.frame}_one_ball_one_dcf.jpg"
if hit.found:cv2.circle(img,(round(hit.x),round(hit.y)),12,(0,0,255),2)
cv2.imwrite(str(fn),img)
print(f"cells_scanned={len(cells)}\nball_found={hit.found}\nscore={hit.score:.6f}\nx={hit.x}\ny={hit.y}\ninside_internal_mesh={None if hit.cell is None else hit.cell.inside}\nz0={a.z0}\nOUT={final_out}\nimage={fn}")
