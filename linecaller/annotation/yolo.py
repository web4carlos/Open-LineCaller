from pathlib import Path
class YoloLabelWriter:
    CLASS_ID=0
    @staticmethod
    def normalize(annotation,image_width,image_height):
        if image_width<=0 or image_height<=0: raise ValueError("Image size must be positive.")
        x=min(max(annotation.x_px,0.0),image_width-1.0);y=min(max(annotation.y_px,0.0),image_height-1.0)
        bw=min(max(2,int(annotation.box_width_px)),image_width);bh=min(max(2,int(annotation.box_height_px)),image_height)
        return x/image_width,y/image_height,bw/image_width,bh/image_height
    @classmethod
    def line(cls,a,w,h):
        x,y,bw,bh=cls.normalize(a,w,h);return f"{cls.CLASS_ID} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
    @classmethod
    def write_positive(cls,path,a,w,h):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(cls.line(a,w,h),encoding="utf-8")
    @staticmethod
    def write_negative(path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("",encoding="utf-8")
