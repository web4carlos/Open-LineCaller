from .models import CellIndex, DCFConfig, FieldRegion

class DynamicCourtField:
    def __init__(self, config=None):
        self.config=config or DCFConfig()
        c=self.config
        self.x0=c.margin_cells
        self.x1=self.x0+c.court_x_cells-1
        self.y0=c.margin_cells
        self.y1=self.y0+c.court_y_cells-1
        self.total_x=c.court_x_cells+2*c.margin_cells
        self.total_y=c.court_y_cells+2*c.margin_cells
        self.total_z=c.z_cells
        self._lit=set()

    @property
    def illuminated(self):
        return frozenset(self._lit)

    def valid(self,p):
        return (0<=p.x<self.total_x and
                0<=p.y<self.total_y and
                0<=p.z<self.total_z)

    def region_at_xy(self,x,y):
        lo=x<self.x0; ro=x>self.x1
        no=y<self.y0; fo=y>self.y1
        if (lo or ro) and (no or fo): return FieldRegion.OUT_CORNER
        if lo: return FieldRegion.OUT_LEFT
        if ro: return FieldRegion.OUT_RIGHT
        if no: return FieldRegion.OUT_NEAR
        if fo: return FieldRegion.OUT_FAR
        b=self.config.boundary_cells
        if (x<self.x0+b or x>self.x1-b or
            y<self.y0+b or y>self.y1-b):
            return FieldRegion.BOUNDARY
        return FieldRegion.INSIDE

    def contact_region(self,p):
        if not self.valid(p) or p.z != 0:
            return None
        return self.region_at_xy(p.x,p.y)

    def illuminate_prediction(self,p,v,uncertainty=0.0,forward_bias=2):
        u=max(0.0,min(1.0,float(uncertainty)))
        c=self.config
        r=round(c.active_radius_xy+u*(c.max_active_radius-c.active_radius_xy))
        rz=round(c.active_radius_z+u*(c.max_active_radius-c.active_radius_z))
        vx,vy,vz=v
        cx=round(p.x+vx); cy=round(p.y+vy); cz=round(p.z+vz)
        sx=(vx>0)-(vx<0); sy=(vy>0)-(vy<0); sz=(vz>0)-(vz<0)
        out=set()

        def addbox(ax,ay,az,rr,rrz):
            for dx in range(-rr,rr+1):
                for dy in range(-rr,rr+1):
                    for dz in range(-rrz,rrz+1):
                        q=CellIndex(ax+dx,ay+dy,az+dz)
                        if self.valid(q):
                            out.add(q)

        addbox(cx,cy,cz,r,rz)
        for s in range(1,max(0,int(forward_bias))+1):
            addbox(cx+sx*s,cy+sy*s,cz+sz*s,
                   max(1,r-s),max(1,rz-s))

        self._lit=out
        return self.illuminated

    def external_illuminated(self):
        return frozenset(
            p for p in self._lit
            if self.region_at_xy(p.x,p.y) != FieldRegion.INSIDE
        )

    def contact_candidates(self):
        return frozenset(p for p in self._lit if p.z == 0)
