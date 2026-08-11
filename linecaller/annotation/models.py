from dataclasses import dataclass
@dataclass(frozen=True)
class BallAnnotation:
    frame_number:int
    x_px:float
    y_px:float
    box_size_px:int=24
@dataclass
class AnnotationStats:
    visited:int=0
    positives:int=0
    negatives:int=0
    @property
    def labeled(self): return self.positives+self.negatives
