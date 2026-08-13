from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import TeamSide


class RallyOutcome(str, Enum):
    RALLY_CONTINUES = "RALLY_CONTINUES"
    RALLY_WON = "RALLY_WON"
    REVIEW = "REVIEW"


class FaultType(str, Enum):
    BALL_OUT = "BALL_OUT"
    SERVE_FAULT = "SERVE_FAULT"
    TWO_BOUNCE_FAULT = "TWO_BOUNCE_FAULT"
    NVZ_FAULT = "NVZ_FAULT"
    DOUBLE_BOUNCE = "DOUBLE_BOUNCE"
    OTHER_FAULT = "OTHER_FAULT"


@dataclass(frozen=True)
class RallyOutcomeContext:
    acting_team: TeamSide
    opponent_team: TeamSide
    decision: str = ""
    fault_type: FaultType | None = None
    fault_team: TeamSide = TeamSide.UNKNOWN
    upstream_review: bool = False


@dataclass(frozen=True)
class RallyOutcomeResult:
    outcome: RallyOutcome
    winner: TeamSide
    loser: TeamSide
    fault_team: TeamSide
    reason: str


class RallyOutcomeEngine:
    """
    CP-0032.3 — Rally Outcome + Fault Ownership.

    Converts a trusted terminal rally event into winner/loser ownership.

    Safety invariant:
    REVIEW uncertainty is monotonic. This engine cannot manufacture a winner
    when upstream evidence/rules require review.
    """

    @staticmethod
    def _opponent(team: TeamSide, a: TeamSide, b: TeamSide) -> TeamSide:
        if team == a:
            return b
        if team == b:
            return a
        return TeamSide.UNKNOWN

    def evaluate(self, ctx: RallyOutcomeContext) -> RallyOutcomeResult:
        decision = str(ctx.decision).upper()

        if ctx.upstream_review or decision == "REVIEW":
            return RallyOutcomeResult(
                RallyOutcome.REVIEW,
                TeamSide.UNKNOWN,
                TeamSide.UNKNOWN,
                TeamSide.UNKNOWN,
                "UPSTREAM_REVIEW_PROPAGATED",
            )

        # Explicit fault ownership has priority.
        if ctx.fault_type is not None:
            fault_team = ctx.fault_team

            if fault_team == TeamSide.UNKNOWN:
                return RallyOutcomeResult(
                    RallyOutcome.REVIEW,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    "FAULT_TEAM_UNKNOWN",
                )

            winner = self._opponent(
                fault_team,
                ctx.acting_team,
                ctx.opponent_team,
            )

            if winner == TeamSide.UNKNOWN:
                return RallyOutcomeResult(
                    RallyOutcome.REVIEW,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    fault_team,
                    "FAULT_TEAM_NOT_IN_RALLY_CONTEXT",
                )

            return RallyOutcomeResult(
                RallyOutcome.RALLY_WON,
                winner,
                fault_team,
                fault_team,
                f"{ctx.fault_type.value}_BY_{fault_team.value}",
            )

        # A trusted OUT call means the team that last played the ball loses.
        if decision == "OUT":
            if ctx.acting_team == TeamSide.UNKNOWN:
                return RallyOutcomeResult(
                    RallyOutcome.REVIEW,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    "ACTING_TEAM_UNKNOWN_FOR_OUT_CALL",
                )

            winner = self._opponent(
                ctx.acting_team,
                ctx.acting_team,
                ctx.opponent_team,
            )

            if winner == TeamSide.UNKNOWN:
                return RallyOutcomeResult(
                    RallyOutcome.REVIEW,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    TeamSide.UNKNOWN,
                    "OPPONENT_TEAM_UNKNOWN_FOR_OUT_CALL",
                )

            return RallyOutcomeResult(
                RallyOutcome.RALLY_WON,
                winner,
                ctx.acting_team,
                ctx.acting_team,
                "BALL_OUT_BY_ACTING_TEAM",
            )

        if decision == "IN" or decision == "":
            return RallyOutcomeResult(
                RallyOutcome.RALLY_CONTINUES,
                TeamSide.UNKNOWN,
                TeamSide.UNKNOWN,
                TeamSide.UNKNOWN,
                "NO_TERMINAL_RALLY_EVENT",
            )

        return RallyOutcomeResult(
            RallyOutcome.REVIEW,
            TeamSide.UNKNOWN,
            TeamSide.UNKNOWN,
            TeamSide.UNKNOWN,
            "UNKNOWN_TERMINAL_DECISION",
        )
