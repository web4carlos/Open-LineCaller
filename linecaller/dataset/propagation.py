from __future__ import annotations

from linecaller.dataset.models import BallBox


class AnnotationPropagator:
    """
    Writes predicted boxes only into unconfirmed frames.
    """

    def propagate(
        self,
        *,
        session,
        predictions,
        visible=True,
        occluded=False,
    ) -> int:
        written = 0

        for frame_number, proposal in predictions:
            if proposal is None:
                continue

            existing = session.frame_annotation(frame_number)

            if existing is not None and existing.ball is not None:
                continue

            box = BallBox(
                x=proposal.x,
                y=proposal.y,
                width=proposal.width,
                height=proposal.height,
            )

            session.set_ball_box(
                frame_number,
                box,
                visible=visible,
                occluded=occluded,
            )

            written += 1

        return written
