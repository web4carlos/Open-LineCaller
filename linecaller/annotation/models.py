from dataclasses import dataclass
@dataclass(frozen=True)
class BallAnnotation:
    frame_number:int
    x_px:float
    y_px:float
    box_width_px:int=24
    box_height_px:int=24
    @property
    def box_size_px(self): return max(int(self.box_width_px),int(self.box_height_px))
@dataclass
class AnnotationStats:
    visited:int=0
    positives:int=0
    negatives:int=0
    @property
    def labeled(self): return self.positives+self.negatives
