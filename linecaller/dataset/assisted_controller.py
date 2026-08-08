from __future__ import annotations

from dataclasses import dataclass, asdict

from linecaller.dataset.models import BallBox
from linecaller.dataset.studio_session import DatasetStudioSession
from linecaller.proposals.dataset_adapter import proposal_to_ball_box
from linecaller.proposals.models import BallProposal
from linecaller.proposals.temporal_tracker import TemporalProposalTracker


@dataclass
class AssistedMetrics:
    proposals_requested: int = 0
    proposals_shown: int = 0
    accepted: int = 0
    adjusted: int = 0
    rejected: int = 0
    auto_advanced: int = 0

    @property
    def acceptance_rate(self) -> float:
        denom = self.accepted + self.adjusted + self.rejected
        if denom == 0:
            return 0.0
        return (self.accepted + self.adjusted) / denom

    def to_dict(self):
        data = asdict(self)
        data["acceptance_rate"] = self.acceptance_rate
        return data


class AssistedAnnotationController:
    def __init__(
        self,
        session: DatasetStudioSession,
        *,
        engine=None,
        temporal_tracker=None,
    ):
        self.session = session
        self.engine = engine
        self.temporal_tracker = temporal_tracker or TemporalProposalTracker()
        self.current_proposal: BallProposal | None = None
        self.metrics = AssistedMetrics()

    def clear_proposal(self):
        self.current_proposal = None

    def request_proposal(self, frame_number, frame):
        self.metrics.proposals_requested += 1

        # Never overwrite a confirmed annotation.
        existing = self.session.frame_annotation(frame_number)
        if existing is not None and existing.ball is not None:
            self.current_proposal = None
            return None

        proposal = None

        if self.engine is not None:
            result = self.engine.propose(frame_number, frame)
            if result.proposals:
                proposal = result.proposals[0]

        if proposal is None:
            proposal = self.temporal_tracker.predict(frame_number)

        self.current_proposal = proposal

        if proposal is not None:
            self.metrics.proposals_shown += 1

        return proposal

    def accept_current(
        self,
        *,
        visible=True,
        occluded=False,
    ) -> BallBox | None:
        if self.current_proposal is None:
            return None

        p = self.current_proposal
        box = proposal_to_ball_box(p)

        self.session.set_ball_box(
            p.frame_number,
            box,
            visible=visible,
            occluded=occluded,
        )

        self.temporal_tracker.observe(
            p.frame_number,
            box.x,
            box.y,
            box.width,
            box.height,
        )

        self.metrics.accepted += 1
        self.current_proposal = None
        return box

    def adjust_current(
        self,
        frame_number: int,
        box: BallBox,
        *,
        visible=True,
        occluded=False,
    ):
        self.session.set_ball_box(
            frame_number,
            box,
            visible=visible,
            occluded=occluded,
        )

        self.temporal_tracker.observe(
            frame_number,
            box.x,
            box.y,
            box.width,
            box.height,
        )

        self.metrics.adjusted += 1
        self.current_proposal = None

    def reject_current(self):
        if self.current_proposal is None:
            return False

        self.current_proposal = None
        self.metrics.rejected += 1
        return True
