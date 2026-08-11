import json
from pathlib import Path
from .models import AnnotationStats
class AnnotationSession:
    VERSION=2
    def __init__(self,video_path,dataset_root):
        self.video_path=str(video_path);self.dataset_root=str(dataset_root);self.current_frame=0;self.box_size_px=24;self.labels={}
    @property
    def session_path(self): return Path(self.dataset_root)/(Path(self.video_path).stem+".annotation.json")
    def mark_positive(self,frame_number,x,y,box_width_px=None,box_height_px=None,score=None,method=None):
        bw=int(box_width_px if box_width_px is not None else self.box_size_px);bh=int(box_height_px if box_height_px is not None else self.box_size_px)
        self.labels[str(int(frame_number))]={"type":"positive","x":float(x),"y":float(y),"box_width_px":bw,"box_height_px":bh,"box_size_px":max(bw,bh),"score":None if score is None else float(score),"method":method}
    def mark_negative(self,frame_number): self.labels[str(int(frame_number))]={"type":"negative"}
    def remove(self,frame_number): self.labels.pop(str(int(frame_number)),None)
    def annotation_for(self,frame_number): return self.labels.get(str(int(frame_number)))
    def stats(self):
        p=sum(1 for x in self.labels.values() if x.get("type")=="positive");n=sum(1 for x in self.labels.values() if x.get("type")=="negative")
        return AnnotationStats(len(self.labels),p,n)
    def save(self):
        p=self.session_path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"version":self.VERSION,"video_path":self.video_path,"dataset_root":self.dataset_root,"current_frame":self.current_frame,"box_size_px":self.box_size_px,"labels":self.labels},indent=2),encoding="utf-8")
    @classmethod
    def load_or_create(cls,video_path,dataset_root):
        s=cls(video_path,dataset_root);p=s.session_path
        if not p.exists(): return s
        d=json.loads(p.read_text(encoding="utf-8"));s.current_frame=int(d.get("current_frame",0));s.box_size_px=int(d.get("box_size_px",24));s.labels=dict(d.get("labels",{}))
        for item in s.labels.values():
            if item.get("type")=="positive":
                old=int(item.get("box_size_px",24));item.setdefault("box_width_px",old);item.setdefault("box_height_px",old)
        return s
