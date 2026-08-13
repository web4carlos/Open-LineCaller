from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from linecaller.calibration_center.precision_refinement import EditableCourt

BOUNDARIES=[
    ("NEAR_BASELINE",0,1),
    ("RIGHT_SIDELINE",1,2),
    ("FAR_BASELINE",2,3),
    ("LEFT_SIDELINE",3,0),
]

class Editor:
    def __init__(self,frame,data,output):
        self.frame=frame
        self.data=data
        self.output=Path(output)
        self.court=EditableCourt.create(data["image_points"],"AUTO")
        self.undo=[]
        self.drag=None
        self.drag_start=None
        self.before=None
        self.message="AUTO detected. Drag corners or outer lines. ENTER=accept"
        self.window="Open-LineCaller Precision Calibration"
        self.zoom="Precision Magnifier"

    def snapshot(self):
        return [p[:] for p in self.court.points]

    def restore(self,p):
        self.court.points=[x[:] for x in p]

    @staticmethod
    def dist_seg(px,py,a,b):
        p=np.array([px,py],float); a=np.array(a,float); b=np.array(b,float)
        v=b-a
        d=float(v@v)
        if d<=1e-9:return float(np.linalg.norm(p-a))
        t=max(0,min(1,float((p-a)@v/d)))
        return float(np.linalg.norm(p-(a+t*v)))

    def hit(self,x,y):
        # Corners first, intentionally large handles.
        for i,p in enumerate(self.court.points):
            if np.hypot(x-p[0],y-p[1]) <= 18:
                return ("corner",i)
        best=None
        for name,i,j in BOUNDARIES:
            d=self.dist_seg(x,y,self.court.points[i],self.court.points[j])
            if d <= 12 and (best is None or d<best[0]):
                best=(d,name)
        return ("line",best[1]) if best else None

    def mouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            h=self.hit(x,y)
            if h:
                self.drag=h
                self.drag_start=(x,y)
                self.before=self.snapshot()
        elif event==cv2.EVENT_MOUSEMOVE and self.drag:
            dx=x-self.drag_start[0]; dy=y-self.drag_start[1]
            self.restore(self.before)
            if self.drag[0]=="corner":
                self.court.move_corner(self.drag[1],x,y)
            else:
                self.court.move_boundary(self.drag[1],dx,dy)
        elif event==cv2.EVENT_LBUTTONUP and self.drag:
            if self.court.valid():
                self.undo.append(self.before)
                self.message="USER REFINED - ENTER=accept, U=undo, R=reset"
            else:
                self.restore(self.before)
                self.message="Invalid court geometry rejected."
            self.drag=None
            self.before=None

    def homography(self):
        src=np.array([[0.,44.],[20.,44.],[20.,0.],[0.,0.]],np.float32)
        dst=np.asarray(self.court.points,np.float32)
        return cv2.getPerspectiveTransform(src,dst)

    def proj(self,pts):
        a=np.array([[[float(x),float(y)]] for x,y in pts],np.float32)
        z=cv2.perspectiveTransform(a,self.homography())
        return [(int(round(p[0][0])),int(round(p[0][1]))) for p in z]

    def draw_line(self,img,a,b,label=None):
        p1,p2=self.proj([a,b])
        cv2.line(img,p1,p2,(255,255,0),2,cv2.LINE_AA)
        if label:
            m=((p1[0]+p2[0])//2,(p1[1]+p2[1])//2)
            cv2.putText(img,label,(m[0]+5,m[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)

    def render(self):
        img=self.frame.copy()
        for a,b,label in [
            ((0,0),(20,0),"FAR BASELINE"),
            ((0,44),(20,44),"NEAR BASELINE"),
            ((0,0),(0,44),"LEFT"),
            ((20,0),(20,44),"RIGHT"),
            ((0,15),(20,15),"FAR NVZ"),
            ((0,22),(20,22),"NET"),
            ((0,29),(20,29),"NEAR NVZ"),
            ((10,0),(10,15),"FAR CENTER"),
            ((10,29),(10,44),"NEAR CENTER"),
        ]: self.draw_line(img,a,b,label)

        for i,(x,y) in enumerate(self.court.points):
            cv2.circle(img,(int(x),int(y)),10,(0,255,255),-1,cv2.LINE_AA)
            cv2.circle(img,(int(x),int(y)),15,(0,0,0),2,cv2.LINE_AA)

        cv2.rectangle(img,(0,0),(img.shape[1],42),(0,0,0),-1)
        cv2.putText(img,self.message,(14,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
        return img

    def save(self):
        out=dict(self.data)
        out["image_points"]=[[float(x),float(y)] for x,y in self.court.points]
        out["calibration_mode"]="AUTO_USER_REFINED"
        out["user_refined"]=True
        out["auto_image_points"]=[[float(x),float(y)] for x,y in self.court.original_points]
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
        print(f"saved={self.output}")

    def run(self):
        cv2.namedWindow(self.window,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window,self.mouse)
        while True:
            cv2.imshow(self.window,self.render())
            k=cv2.waitKey(20)&0xFF
            if k in (13,10):
                if self.court.valid():
                    self.save(); break
            elif k in (ord("u"),ord("U")):
                if self.undo:self.restore(self.undo.pop())
            elif k in (ord("r"),ord("R")):
                self.undo.append(self.snapshot()); self.court.reset()
                self.message="Reset to AUTO result."
            elif k==27:
                print("cancelled=True"); break
        cv2.destroyAllWindows()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True)
    p.add_argument("--calibration",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--frame",type=int,default=100)
    a=p.parse_args()
    data=json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    cap=cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame)
    ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit("Could not read frame.")
    Editor(frame,data,a.output).run()

if __name__=="__main__":main()
