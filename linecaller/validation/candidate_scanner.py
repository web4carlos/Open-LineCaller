import cv2
from linecaller.live.pipeline_factory import create_live_pipeline_adapter
from .annotation_models import AnnotationCandidate

class CandidateScanner:
    def __init__(self, *, pipeline=None, min_frame_gap=3):
        self.pipeline = pipeline or create_live_pipeline_adapter()
        self.min_frame_gap = int(min_frame_gap)

    def scan(self, video_path, *, progress_callback=None):
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_number=0; last=-10**9; out=[]
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                result=self.pipeline.process_frame(
                    frame_number=frame_number,
                    frame=frame,
                    timestamp=frame_number/fps,
                )
                event=result.event
                if event is not None and frame_number-last >= self.min_frame_gap:
                    r=result.result
                    out.append(AnnotationCandidate(
                        frame=frame_number,
                        confidence=float(getattr(event,"confidence",0.0)),
                        x=getattr(r,"ball_x",None),
                        y=getattr(r,"ball_y",None),
                        suggested_decision=getattr(getattr(event,"decision",None),"value",None),
                        metadata=getattr(event,"metadata",None),
                    ))
                    last=frame_number
                if progress_callback:
                    progress_callback(frame_number, frame_count)
                frame_number += 1
        finally:
            cap.release()
        return tuple(out)
