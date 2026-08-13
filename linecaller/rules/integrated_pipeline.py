from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import TeamSide
from .rally_outcome import (
    FaultType,
    RallyOutcome,
    RallyOutcomeContext,
    RallyOutcomeEngine,
)
from .serve_two_bounce import (
    ServeContext,
    ServeLegalityEvaluator,
    ServeRuling,
    TwoBounceRule,
    TwoBounceState,
)
from .nvz import NVZContext, NVZRuleEvaluator, NVZRuling


class IntegratedRulesStatus(str, Enum):
    CONTINUE = "CONTINUE"
    RALLY_WON = "RALLY_WON"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class IntegratedRulesResult:
    status: IntegratedRulesStatus
    winner: TeamSide = TeamSide.UNKNOWN
    loser: TeamSide = TeamSide.UNKNOWN
    reason: str = ""


class IntegratedPickleballRulesPipeline:
    """
    CP-0032.4 — Integrated Pickleball Rules Pipeline.

    Orchestrates the deterministic CP-0032 rule components and preserves the
    most important safety invariant:

        REVIEW is monotonic.

    Once an event requires REVIEW, this pipeline does not manufacture a
    definitive rally winner from that event.
    """

    def __init__(self):
        self.serve = ServeLegalityEvaluator()
        self.two_bounce = TwoBounceRule()
        self.nvz = NVZRuleEvaluator()
        self.outcome = RallyOutcomeEngine()

    @staticmethod
    def _review(reason: str) -> IntegratedRulesResult:
        return IntegratedRulesResult(
            IntegratedRulesStatus.REVIEW,
            reason=reason,
        )

    @staticmethod
    def _continue(reason: str) -> IntegratedRulesResult:
        return IntegratedRulesResult(
            IntegratedRulesStatus.CONTINUE,
            reason=reason,
        )

    @staticmethod
    def _from_outcome(result) -> IntegratedRulesResult:
        if result.outcome == RallyOutcome.REVIEW:
            return IntegratedRulesResult(
                IntegratedRulesStatus.REVIEW,
                reason=result.reason,
            )
        if result.outcome == RallyOutcome.RALLY_WON:
            return IntegratedRulesResult(
                IntegratedRulesStatus.RALLY_WON,
                winner=result.winner,
                loser=result.loser,
                reason=result.reason,
            )
        return IntegratedRulesResult(
            IntegratedRulesStatus.CONTINUE,
            reason=result.reason,
        )

    def reset_rally(self):
        self.two_bounce.reset()

    def evaluate_serve(
        self,
        *,
        serving_team: TeamSide,
        receiving_team: TeamSide,
        server_side: str,
        receiving_zone: str,
        nearest_line: str = "",
        on_line: bool = False,
        decision: str = "IN",
    ) -> IntegratedRulesResult:
        s = self.serve.evaluate(
            ServeContext(
                serving_team=serving_team.value,
                server_side=server_side,
                receiving_zone=receiving_zone,
                nearest_line=nearest_line,
                on_line=on_line,
                upstream_decision=decision,
            )
        )

        if s.ruling == ServeRuling.REVIEW:
            return self._review(s.reason)

        if s.ruling == ServeRuling.SERVE_FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=serving_team,
                    opponent_team=receiving_team,
                    fault_type=FaultType.SERVE_FAULT,
                    fault_team=serving_team,
                )
            )
            return self._from_outcome(o)

        t = self.two_bounce.receiver_bounce(decision)
        if t.state == TwoBounceState.REVIEW:
            return self._review(t.reason)
        if t.state == TwoBounceState.FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=serving_team,
                    opponent_team=receiving_team,
                    fault_type=FaultType.SERVE_FAULT,
                    fault_team=serving_team,
                )
            )
            return self._from_outcome(o)

        return self._continue("LEGAL_SERVE_RECEIVER_BOUNCE_CONFIRMED")

    def evaluate_return(
        self,
        *,
        receiving_team: TeamSide,
        serving_team: TeamSide,
        return_volley: bool,
    ) -> IntegratedRulesResult:
        r = self.two_bounce.receiver_return_hit(return_volley)

        if r.state == TwoBounceState.FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=receiving_team,
                    opponent_team=serving_team,
                    fault_type=FaultType.TWO_BOUNCE_FAULT,
                    fault_team=receiving_team,
                )
            )
            return self._from_outcome(o)

        if r.state == TwoBounceState.REVIEW:
            return self._review(r.reason)

        return self._continue(r.reason)

    def evaluate_server_side_bounce(
        self,
        *,
        serving_team: TeamSide,
        receiving_team: TeamSide,
        decision: str,
    ) -> IntegratedRulesResult:
        r = self.two_bounce.server_side_bounce(decision)

        if r.state == TwoBounceState.REVIEW:
            return self._review(r.reason)

        if r.state == TwoBounceState.FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=receiving_team,
                    opponent_team=serving_team,
                    decision="OUT",
                )
            )
            return self._from_outcome(o)

        return self._continue(r.reason)

    def evaluate_server_volley_before_required_bounce(
        self,
        *,
        serving_team: TeamSide,
        receiving_team: TeamSide,
    ) -> IntegratedRulesResult:
        r = self.two_bounce.server_hit_before_bounce()

        if r.state == TwoBounceState.FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=serving_team,
                    opponent_team=receiving_team,
                    fault_type=FaultType.TWO_BOUNCE_FAULT,
                    fault_team=serving_team,
                )
            )
            return self._from_outcome(o)

        return self._continue(r.reason)

    def evaluate_nvz_contact(
        self,
        *,
        acting_team: TeamSide,
        opponent_team: TeamSide,
        volley: bool,
        player_in_nvz: bool = False,
        entered_nvz_after_volley: bool = False,
        momentum_complete: bool = True,
        decision: str = "IN",
    ) -> IntegratedRulesResult:
        n = self.nvz.evaluate(
            NVZContext(
                volley=volley,
                player_in_nvz=player_in_nvz,
                entered_nvz_after_volley=entered_nvz_after_volley,
                momentum_complete=momentum_complete,
                upstream_decision=decision,
            )
        )

        if n.ruling == NVZRuling.REVIEW:
            return self._review(n.reason)

        if n.ruling == NVZRuling.NVZ_FAULT:
            o = self.outcome.evaluate(
                RallyOutcomeContext(
                    acting_team=acting_team,
                    opponent_team=opponent_team,
                    fault_type=FaultType.NVZ_FAULT,
                    fault_team=acting_team,
                )
            )
            return self._from_outcome(o)

        return self._continue(n.reason)

    def evaluate_ball_decision(
        self,
        *,
        acting_team: TeamSide,
        opponent_team: TeamSide,
        decision: str,
    ) -> IntegratedRulesResult:
        o = self.outcome.evaluate(
            RallyOutcomeContext(
                acting_team=acting_team,
                opponent_team=opponent_team,
                decision=decision,
            )
        )
        return self._from_outcome(o)
