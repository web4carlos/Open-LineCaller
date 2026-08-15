import argparse,json,cv2
from pathlib import Path
from linecaller.lattice import PerspectiveCourtMesh,MeshConfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--frame",type=int,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--lateral-cells",type=int,default=20)
    p.add_argument("--depth-cells",type=int,default=44)
    p.add_argument("--height-cells",type=int,default=12)
    p.add_argument("--near-height-px",type=float,default=650)
    p.add_argument("--gamma",type=float,default=1.35)
    a=p.parse_args()
    cal=json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    cap=cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit("Cannot read frame")
    mesh=PerspectiveCourtMesh(cal["image_points"],MeshConfig(a.lateral_cells,a.depth_cells,a.height_cells,a.near_height_px,a.gamma))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out),mesh.draw(frame)): raise SystemExit("Could not write output")
    print(out)

if __name__=="__main__": main()
