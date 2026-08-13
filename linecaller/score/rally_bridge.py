from __future__ import annotations

from dataclasses import dataclass

from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import RallyOutcome, RallyOutcomeResult

from .engine import ScoreState, ScoreUpdate, SideOutScoreEngine


@dataclass(frozen=True)
class RallyScoreBridgeResult:
    state: ScoreState
    score_changed: bool
    action: str
    reason: str


class RallyOutcomeScoreBridge:
    """
    CP-0033.1 — Bridge trusted CP-0032 rally outcomes into CP-0033 scoring.

    Safety contract:
    - RALLY_WON with a known winner may update score/service state.
    - REVIEW never changes score/service state.
    - RALLY_CONTINUES never changes score/service state.
    - malformed/unknown winners fail safe and do not change score.
    """

    def __init__(self, score_engine: SideOutScoreEngine):
        self.score_engine = score_engine

    def apply(self, outcome: RallyOutcomeResult) -> RallyScoreBridgeResult:
        before = self.score_engine.state

        if outcome.outcome == RallyOutcome.REVIEW:
            return RallyScoreBridgeResult(
                before,
                False,
                "NO_CHANGE",
                "REVIEW_CANNOT_CHANGE_SCORE",
            )

        if outcome.outcome == RallyOutcome.RALLY_CONTINUES:
            return RallyScoreBridgeResult(
                before,
                False,
                "NO_CHANGE",
                "RALLY_NOT_TERMINAL",
            )

        if outcome.outcome != RallyOutcome.RALLY_WON:
            return RallyScoreBridgeResult(
                before,
                False,
                "NO_CHANGE",
                "UNKNOWN_RALLY_OUTCOME",
            )

        if outcome.winner not in (TeamSide.A, TeamSide.B):
            return RallyScoreBridgeResult(
                before,
                False,
                "NO_CHANGE",
                "RALLY_WINNER_UNKNOWN",
            )

        update: ScoreUpdate = self.score_engine.apply_rally_winner(
            outcome.winner
        )

        after = update.state
        changed = after != before

        return RallyScoreBridgeResult(
            after,
            changed,
            update.action,
            update.reason,
        )
