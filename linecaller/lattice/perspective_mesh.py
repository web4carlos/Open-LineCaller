from dataclasses import dataclass
import cv2, numpy as np

@dataclass(frozen=True)
class MeshConfig:
    lateral_cells:int=20
    depth_cells:int=44
    height_cells:int=12
    near_height_px:float=650.0
    gamma:float=1.35

class PerspectiveCourtMesh:
    def __init__(self,image_points,config=None):
        self.c=config or MeshConfig()
        pts=np.asarray(image_points,dtype=float)
        if pts.shape!=(4,2): raise ValueError("image_points must be 4x2")
        self.nl,self.nr,self.fr,self.fl=pts

    def _lerp(self,a,b,t): return a*(1-t)+b*t

    def floor_edges(self,t):
        return self._lerp(self.nl,self.fl,t), self._lerp(self.nr,self.fr,t)

    def floor_point(self,u,t):
        l,r=self.floor_edges(t)
        return self._lerp(l,r,u)

    def height_px(self,t):
        t=max(0.0,min(1.0,float(t)))
        return self.c.near_height_px*((1-t)**self.c.gamma)

    def voxel_point(self,u,t,h):
        p=self.floor_point(u,t).copy()
        p[1]-=self.height_px(t)*max(0.0,min(1.0,float(h)))
        return p

    def candidate_window(self,previous_cell,velocity_cells,radius=2):
        px,py,pz=previous_cell; vx,vy,vz=velocity_cells
        cx,cy,cz=round(px+vx),round(py+vy),round(pz+vz)
        out=set()
        for x in range(cx-radius,cx+radius+1):
            for y in range(cy-radius,cy+radius+1):
                for z in range(cz-radius,cz+radius+1):
                    if 0<=x<self.c.lateral_cells and 0<=y<self.c.depth_cells and 0<=z<self.c.height_cells:
                        out.add((x,y,z))
        return out

    @staticmethod
    def _pt(p): return int(round(float(p[0]))),int(round(float(p[1])))

    def draw(self,frame):
        out=frame.copy(); c=self.c
        # floor grid
        for j in range(c.depth_cells+1):
            t=j/c.depth_cells; l,r=self.floor_edges(t)
            cv2.line(out,self._pt(l),self._pt(r),(120,220,120),1,cv2.LINE_AA)
        for i in range(c.lateral_cells+1):
            u=i/c.lateral_cells
            cv2.line(out,self._pt(self.floor_point(u,0)),self._pt(self.floor_point(u,1)),(120,220,120),1,cv2.LINE_AA)
        # wedge verticals
        dstep=max(1,c.depth_cells//11); lstep=max(1,c.lateral_cells//10)
        for j in range(0,c.depth_cells+1,dstep):
            t=j/c.depth_cells
            for i in range(0,c.lateral_cells+1,lstep):
                u=i/c.lateral_cells
                cv2.line(out,self._pt(self.voxel_point(u,t,0)),self._pt(self.voxel_point(u,t,1)),(100,180,255),1,cv2.LINE_AA)
        # upper slices
        hstep=max(1,c.height_cells//6)
        for k in range(hstep,c.height_cells+1,hstep):
            h=k/c.height_cells
            for j in range(0,c.depth_cells+1,dstep):
                t=j/c.depth_cells
                cv2.line(out,self._pt(self.voxel_point(0,t,h)),self._pt(self.voxel_point(1,t,h)),(255,180,90),1,cv2.LINE_AA)
        return out
