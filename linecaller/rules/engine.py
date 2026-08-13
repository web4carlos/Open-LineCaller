from __future__ import annotations
from dataclasses import replace

from .models import (
    RallyEvent,
    RallyEventType,
    RallyPhase,
    RallyState,
    RulesResult,
    TeamSide,
)


class PickleballRulesEngine:
    """
    CP-0032.0 — Rules Engine Foundation.

    This layer interprets trusted officiating events.
    It does NOT override upstream REVIEW uncertainty.

    CP-0032.0 establishes rally lifecycle/state only.
    Serve legality, two-bounce rule, NVZ faults and scoring are added
    in subsequent CP-0032.x capabilities.
    """

    def __init__(self):
        self.state = RallyState()

    def reset(self):
        self.state = RallyState()
        return self.state

    def apply(self, event: RallyEvent) -> RulesResult:
        s = self.state

        if s.last_frame >= 0 and event.frame < s.last_frame:
            return RulesResult(
                s, False, "IGNORED",
                "EVENT_FRAME_PRECEDES_CURRENT_RALLY_STATE",
            )

        # Uncertainty is monotonic: rules cannot upgrade REVIEW.
        if (
            event.event_type == RallyEventType.REVIEW
            or str(event.decision).upper() == "REVIEW"
        ):
            self.state = replace(
                s,
                phase=RallyPhase.REVIEW,
                review_required=True,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "REVIEW",
                "UPSTREAM_REVIEW_PROPAGATED",
            )

        if event.event_type == RallyEventType.START_RALLY:
            receiving = (
                TeamSide.B if event.team == TeamSide.A
                else TeamSide.A if event.team == TeamSide.B
                else TeamSide.UNKNOWN
            )
            self.state = RallyState(
                phase=RallyPhase.SERVE_FLIGHT,
                serving_team=event.team,
                receiving_team=receiving,
                bounce_count=0,
                hit_count=0,
                rally_active=True,
                review_required=False,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "RALLY_STARTED",
                "SERVE_SEQUENCE_STARTED",
            )

        if not s.rally_active:
            return RulesResult(
                s, False, "IGNORED",
                "NO_ACTIVE_RALLY",
            )

        if event.event_type == RallyEventType.BOUNCE:
            count = s.bounce_count + 1

            if s.phase == RallyPhase.SERVE_FLIGHT:
                phase = RallyPhase.RETURN_FLIGHT
            elif s.phase == RallyPhase.RETURN_FLIGHT:
                phase = RallyPhase.THIRD_SHOT_FLIGHT
            elif s.phase == RallyPhase.THIRD_SHOT_FLIGHT:
                phase = RallyPhase.OPEN_RALLY
            else:
                phase = s.phase

            self.state = replace(
                s,
                phase=phase,
                bounce_count=count,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "BOUNCE_ACCEPTED",
                "RALLY_BOUNCE_RECORDED",
            )

        if event.event_type == RallyEventType.HIT:
            self.state = replace(
                s,
                hit_count=s.hit_count + 1,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "HIT_ACCEPTED",
                "RALLY_HIT_RECORDED",
            )

        if event.event_type == RallyEventType.FAULT:
            self.state = replace(
                s,
                phase=RallyPhase.ENDED,
                rally_active=False,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "FAULT",
                event.reason or "FAULT_EVENT_ACCEPTED",
            )

        if event.event_type == RallyEventType.END_RALLY:
            self.state = replace(
                s,
                phase=RallyPhase.ENDED,
                rally_active=False,
                last_frame=event.frame,
            )
            return RulesResult(
                self.state, True, "RALLY_ENDED",
                event.reason or "END_RALLY_EVENT_ACCEPTED",
            )

        return RulesResult(
            s, False, "IGNORED",
            "UNSUPPORTED_RALLY_EVENT",
        )
