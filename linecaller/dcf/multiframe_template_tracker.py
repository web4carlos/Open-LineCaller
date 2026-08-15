from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from .template_matcher import DCFBallTemplateMatcher


@dataclass(frozen=True)
class TemplateTrackPoint:
    frame: int
    found: bool
    x: float | None
    y: float | None
    score: float
    scale: float | None
    roi: tuple[int,int,int,int]


class DCFMultiFrameTemplateTracker:
    """
    CP-0035.8
    Track the same ball reference across consecutive frames without YOLO.

    The DCF supplies/updates a compact ROI around the last accepted match.
    The matcher searches the reference at multiple scales to accommodate
    perspective-driven apparent size changes.
    """

    def __init__(
        self,
        *,
        min_score: float = 0.36,
        roi_half_width: int = 90,
        roi_half_height: int = 70,
        lost_expand_px: int = 35,
        max_lost_frames: int = 4,
        scales=(0.50,0.65,0.80,0.95,1.0,1.15,1.30,1.50,1.75,2.0),
    ):
        self.matcher = DCFBallTemplateMatcher(min_score=min_score, scales=scales)
        self.roi_half_width = int(roi_half_width)
        self.roi_half_height = int(roi_half_height)
        self.lost_expand_px = int(lost_expand_px)
        self.max_lost_frames = int(max_lost_frames)

    @staticmethod
    def _roi(center_x, center_y, half_w, half_h, frame_shape):
        h,w = frame_shape[:2]
        x1=max(0,int(round(center_x-half_w)))
        y1=max(0,int(round(center_y-half_h)))
        x2=min(w,int(round(center_x+half_w)))
        y2=min(h,int(round(center_y+half_h)))
        return x1,y1,x2,y2

    def track(
        self,
        video_path: str,
        template_path: str,
        *,
        start_frame: int,
        frame_count: int,
        initial_center: tuple[float,float],
    ) -> list[TemplateTrackPoint]:
        template=cv2.imread(template_path)
        if template is None:
            raise RuntimeError(f"Could not read template: {template_path}")

        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

        cx,cy=map(float,initial_center)
        lost=0
        points=[]

        try:
            for i in range(frame_count):
                frame_no=start_frame+i
                ok,frame=cap.read()
                if not ok:
                    break

                expand=lost*self.lost_expand_px
                roi=self._roi(
                    cx,cy,
                    self.roi_half_width+expand,
                    self.roi_half_height+expand,
                    frame.shape,
                )

                result=self.matcher.match(frame,template,roi)

                if result.found:
                    cx=float(result.x)
                    cy=float(result.y)
                    lost=0
                else:
                    lost += 1

                points.append(
                    TemplateTrackPoint(
                        frame=frame_no,
                        found=result.found,
                        x=result.x,
                        y=result.y,
                        score=result.score,
                        scale=result.scale,
                        roi=result.roi,
                    )
                )

                if lost > self.max_lost_frames:
                    break
        finally:
            cap.release()

        return points
