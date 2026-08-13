from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NVZRuling(str, Enum):
    LEGAL = "LEGAL"
    NVZ_FAULT = "NVZ_FAULT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class NVZContext:
    """
    Evidence available for one player/ball contact.

    volley:
        True only when the ball is struck before bouncing.

    player_in_nvz:
        Player is touching the NVZ or its boundary line at ball contact.

    entered_nvz_after_volley:
        Player subsequently touches the NVZ because of the volley action.

    momentum_complete:
        True when the player's momentum from the volley is known to have ended.
        If the evidence window ends before this can be established, REVIEW is
        safer than declaring the play legal.

    upstream_decision:
        Trusted upstream state. REVIEW is never upgraded here.
    """
    volley: bool
    player_in_nvz: bool = False
    entered_nvz_after_volley: bool = False
    momentum_complete: bool = True
    upstream_decision: str = "IN"


@dataclass(frozen=True)
class NVZResult:
    ruling: NVZRuling
    reason: str


class NVZRuleEvaluator:
    """
    CP-0032.2 — Non-Volley Zone / Kitchen rule evaluator.

    Important:
    Merely standing in the NVZ is not a fault.
    The restriction is tied to volleying and volley momentum.
    """

    def evaluate(self, ctx: NVZContext) -> NVZResult:
        if str(ctx.upstream_decision).upper() == "REVIEW":
            return NVZResult(
                NVZRuling.REVIEW,
                "UPSTREAM_REVIEW_PROPAGATED",
            )

        # A groundstroke after a bounce may legally be played from the NVZ.
        if not ctx.volley:
            return NVZResult(
                NVZRuling.LEGAL,
                "BOUNCED_BALL_MAY_BE_PLAYED_FROM_NVZ",
            )

        # Volley while touching NVZ or its boundary.
        if ctx.player_in_nvz:
            return NVZResult(
                NVZRuling.NVZ_FAULT,
                "VOLLEY_WHILE_TOUCHING_NVZ",
            )

        # Momentum caused by the volley carries player into NVZ.
        if ctx.entered_nvz_after_volley:
            return NVZResult(
                NVZRuling.NVZ_FAULT,
                "VOLLEY_MOMENTUM_ENTERED_NVZ",
            )

        # If we cannot observe the end of momentum, do not invent legality.
        if not ctx.momentum_complete:
            return NVZResult(
                NVZRuling.REVIEW,
                "VOLLEY_MOMENTUM_NOT_FULLY_OBSERVED",
            )

        return NVZResult(
            NVZRuling.LEGAL,
            "LEGAL_VOLLEY_OUTSIDE_NVZ",
        )
