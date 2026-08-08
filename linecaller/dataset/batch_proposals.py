from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchProposalItem:
    frame_number: int
    proposal: object | None


class BatchProposalRunner:
    def __init__(self, *, planner=None):
        from .range_planner import RangeProposalPlanner
        self.planner = planner or RangeProposalPlanner()

    def run(
        self,
        *,
        session,
        controller,
        frame_provider,
        start_frame: int,
        end_frame: int,
    ) -> tuple[BatchProposalItem, ...]:

        items = []

        for frame_number in self.planner.eligible_frames(
            session,
            start_frame=start_frame,
            end_frame=end_frame,
        ):
            frame = frame_provider(frame_number)

            if frame is None:
                items.append(
                    BatchProposalItem(
                        frame_number=frame_number,
                        proposal=None,
                    )
                )
                continue

            proposal = controller.request_proposal(
                frame_number,
                frame,
            )

            items.append(
                BatchProposalItem(
                    frame_number=frame_number,
                    proposal=proposal,
                )
            )

            controller.clear_proposal()

        return tuple(items)
