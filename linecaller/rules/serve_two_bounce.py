from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ServeRuling(str, Enum):
    LEGAL_SERVE = "LEGAL_SERVE"
    SERVE_FAULT = "SERVE_FAULT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class ServeContext:
    serving_team: str
    server_side: str  # RIGHT or LEFT
    receiving_zone: str
    nearest_line: str = ""
    on_line: bool = False
    upstream_decision: str = "IN"


@dataclass(frozen=True)
class ServeResult:
    ruling: ServeRuling
    reason: str


class ServeLegalityEvaluator:
    """
    CP-0032.1

    Uses semantic court zones, not raw pixel geometry.

    For a standard doubles/singles full court:
    - server on RIGHT serves diagonally to opponent LEFT service court
    - server on LEFT serves diagonally to opponent RIGHT service court
    - serve must clear NVZ
    - serve on NVZ line is a fault
    - any upstream REVIEW remains REVIEW
    """

    def evaluate(self, ctx: ServeContext) -> ServeResult:
        if str(ctx.upstream_decision).upper() == "REVIEW":
            return ServeResult(ServeRuling.REVIEW, "UPSTREAM_REVIEW_PROPAGATED")

        side = str(ctx.server_side).upper()
        zone = str(ctx.receiving_zone).upper()
        line = str(ctx.nearest_line).upper()

        if "KITCHEN" in zone or "NVZ" in zone:
            return ServeResult(ServeRuling.SERVE_FAULT, "SERVE_LANDED_IN_NOVOLLEY_ZONE")

        if ctx.on_line and "NVZ" in line:
            return ServeResult(ServeRuling.SERVE_FAULT, "SERVE_TOUCHED_NVZ_LINE")

        if side == "RIGHT":
            legal = zone.endswith("LEFT_SERVICE")
        elif side == "LEFT":
            legal = zone.endswith("RIGHT_SERVICE")
        else:
            return ServeResult(ServeRuling.REVIEW, "UNKNOWN_SERVER_SIDE")

        if not legal:
            return ServeResult(ServeRuling.SERVE_FAULT, "SERVE_WRONG_SERVICE_COURT")

        return ServeResult(ServeRuling.LEGAL_SERVE, "SERVE_LANDED_IN_CORRECT_DIAGONAL_COURT")


class TwoBounceState(str, Enum):
    WAIT_RECEIVER_BOUNCE = "WAIT_RECEIVER_BOUNCE"
    WAIT_SERVER_BOUNCE = "WAIT_SERVER_BOUNCE"
    OPEN_RALLY = "OPEN_RALLY"
    FAULT = "FAULT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class TwoBounceResult:
    state: TwoBounceState
    ruling: str
    reason: str


class TwoBounceRule:
    """
    Tracks the mandatory bounce on receiver side, then mandatory bounce on
    serving side before open volley play is allowed.
    """

    def __init__(self):
        self.state = TwoBounceState.WAIT_RECEIVER_BOUNCE

    def reset(self):
        self.state = TwoBounceState.WAIT_RECEIVER_BOUNCE

    def receiver_bounce(self, decision="IN"):
        d = str(decision).upper()
        if d == "REVIEW":
            self.state = TwoBounceState.REVIEW
            return TwoBounceResult(self.state, "REVIEW", "RECEIVER_BOUNCE_REVIEW")
        if d == "OUT":
            self.state = TwoBounceState.FAULT
            return TwoBounceResult(self.state, "FAULT", "SERVE_OUT_OR_INVALID")
        self.state = TwoBounceState.WAIT_SERVER_BOUNCE
        return TwoBounceResult(self.state, "CONTINUE", "RECEIVER_BOUNCE_CONFIRMED")

    def receiver_return_hit(self, volley: bool):
        if self.state != TwoBounceState.WAIT_SERVER_BOUNCE:
            return TwoBounceResult(self.state, "IGNORED", "UNEXPECTED_RETURN_HIT_STATE")
        if volley:
            self.state = TwoBounceState.FAULT
            return TwoBounceResult(self.state, "FAULT", "RECEIVER_VOLLEYED_BEFORE_REQUIRED_BOUNCE")
        return TwoBounceResult(self.state, "CONTINUE", "RECEIVER_RETURN_AFTER_BOUNCE")

    def server_side_bounce(self, decision="IN"):
        if self.state != TwoBounceState.WAIT_SERVER_BOUNCE:
            return TwoBounceResult(self.state, "IGNORED", "UNEXPECTED_SERVER_BOUNCE_STATE")
        d = str(decision).upper()
        if d == "REVIEW":
            self.state = TwoBounceState.REVIEW
            return TwoBounceResult(self.state, "REVIEW", "SERVER_SIDE_BOUNCE_REVIEW")
        if d == "OUT":
            self.state = TwoBounceState.FAULT
            return TwoBounceResult(self.state, "FAULT", "RETURN_OUT")
        self.state = TwoBounceState.OPEN_RALLY
        return TwoBounceResult(self.state, "CONTINUE", "TWO_BOUNCE_RULE_SATISFIED")

    def server_hit_before_bounce(self):
        if self.state == TwoBounceState.WAIT_SERVER_BOUNCE:
            self.state = TwoBounceState.FAULT
            return TwoBounceResult(self.state, "FAULT", "SERVING_TEAM_VOLLEYED_BEFORE_REQUIRED_BOUNCE")
        return TwoBounceResult(self.state, "IGNORED", "NO_TWO_BOUNCE_VIOLATION")
