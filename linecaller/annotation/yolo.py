from pathlib import Path
from .models import BallAnnotation
class YoloLabelWriter:
    CLASS_ID=0
    @staticmethod
    def normalize(annotation,image_width,image_height):
        if image_width<=0 or image_height<=0: raise ValueError("Image size must be positive.")
        box=max(2,int(annotation.box_size_px))
        x=min(max(annotation.x_px,0.0),image_width-1.0)
        y=min(max(annotation.y_px,0.0),image_height-1.0)
        return x/image_width,y/image_height,min(box,image_width)/image_width,min(box,image_height)/image_height
    @classmethod
    def line(cls,annotation,image_width,image_height):
        x,y,w,h=cls.normalize(annotation,image_width,image_height)
        return f"{cls.CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
    @classmethod
    def write_positive(cls,path,annotation,image_width,image_height):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(cls.line(annotation,image_width,image_height),encoding="utf-8")
    @staticmethod
    def write_negative(path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("",encoding="utf-8")
