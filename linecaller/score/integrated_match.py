from __future__ import annotations

from dataclasses import dataclass

from linecaller.rules.models import TeamSide
from linecaller.rules.rally_outcome import RallyOutcomeResult

from .announcer import AnnouncerState, build_announcer_state
from .engine import MatchFormat
from .history import ScoreHistory, ScoreHistoryEntry
from .match import MatchController


@dataclass(frozen=True)
class IntegratedMatchResult:
    action: str
    reason: str
    score_changed: bool
    game_committed: bool
    match_over: bool
    announcer: AnnouncerState
    history_entry: ScoreHistoryEntry


class IntegratedMatchScoringPipeline:
    """
    CP-0033.4 — Integrated Match Scoring Pipeline.

    One high-level entry point for trusted CP-0032 RallyOutcomeResult events.

    Flow:
      RallyOutcomeResult
          -> RallyOutcomeScoreBridge
          -> SideOutScoreEngine
          -> MatchController
          -> ScoreHistory
          -> AnnouncerState

    Safety invariant:
      REVIEW / RALLY_CONTINUES / malformed outcomes never change score.
    """

    def __init__(
        self,
        *,
        best_of: int = 3,
        match_format: MatchFormat = MatchFormat.DOUBLES,
        starting_team: TeamSide = TeamSide.A,
        target_score: int = 11,
        win_by: int = 2,
    ):
        self.match = MatchController(
            best_of=best_of,
            match_format=match_format,
            starting_team=starting_team,
            target_score=target_score,
            win_by=win_by,
        )
        self.history = ScoreHistory()

    def _announcer(self, action: str, reason: str) -> AnnouncerState:
        s = self.match.state
        return build_announcer_state(
            score_state=self.match.score_engine.state,
            game_number=s.current_game,
            games_a=s.games_a,
            games_b=s.games_b,
            last_action=action,
            last_reason=reason,
            match_over=s.match_over,
            match_winner=s.match_winner,
        )

    def _record(self, event: str, action: str, reason: str) -> ScoreHistoryEntry:
        s = self.match.state
        return self.history.record(
            game_number=s.current_game,
            event=event,
            action=action,
            reason=reason,
            score_state=self.match.score_engine.state,
            games_a=s.games_a,
            games_b=s.games_b,
            match_over=s.match_over,
        )

    def apply_rally_outcome(
        self,
        outcome: RallyOutcomeResult,
    ) -> IntegratedMatchResult:
        if self.match.state.match_over:
            action = "NO_CHANGE"
            reason = "MATCH_ALREADY_OVER"
            entry = self._record("RALLY", action, reason)
            return IntegratedMatchResult(
                action=action,
                reason=reason,
                score_changed=False,
                game_committed=False,
                match_over=True,
                announcer=self._announcer(action, reason),
                history_entry=entry,
            )

        bridge = self.match.bridge.apply(outcome)
        action = bridge.action
        reason = bridge.reason
        changed = bridge.score_changed
        committed = False

        # Commit game exactly once when the score engine closes it.
        if self.match.score_engine.state.game_over:
            update = self.match.commit_finished_game()
            committed = update.action in ("NEW_GAME", "MATCH_WON")
            action = update.action
            reason = update.reason

        entry = self._record("RALLY", action, reason)
        announcer = self._announcer(action, reason)

        return IntegratedMatchResult(
            action=action,
            reason=reason,
            score_changed=changed,
            game_committed=committed,
            match_over=self.match.state.match_over,
            announcer=announcer,
            history_entry=entry,
        )
