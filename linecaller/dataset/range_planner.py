from __future__ import annotations


class RangeProposalPlanner:
    def __init__(self, *, max_span: int = 60):
        self.max_span = int(max_span)

    def eligible_frames(
        self,
        session,
        *,
        start_frame: int,
        end_frame: int,
    ) -> tuple[int, ...]:

        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        if end_frame - start_frame > self.max_span:
            end_frame = start_frame + self.max_span

        frames = []

        for frame in range(start_frame, end_frame + 1):
            ann = session.frame_annotation(frame)

            if ann is not None and ann.ball is not None:
                continue

            frames.append(frame)

        return tuple(frames)
