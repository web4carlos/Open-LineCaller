from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from linecaller.rules.models import TeamSide
from .engine import ScoreState


@dataclass(frozen=True)
class ScoreHistoryEntry:
    sequence: int
    game_number: int
    event: str
    action: str
    reason: str
    score_a: int
    score_b: int
    serving_team: TeamSide
    server_number: int
    server_court: str
    games_a: int
    games_b: int
    game_over: bool
    match_over: bool


class ScoreHistory:
    """Append-only audit history for score/match state transitions."""

    def __init__(self):
        self._entries: list[ScoreHistoryEntry] = []

    @property
    def entries(self) -> tuple[ScoreHistoryEntry, ...]:
        return tuple(self._entries)

    @property
    def last(self) -> Optional[ScoreHistoryEntry]:
        return self._entries[-1] if self._entries else None

    def record(
        self,
        *,
        game_number: int,
        event: str,
        action: str,
        reason: str,
        score_state: ScoreState,
        games_a: int,
        games_b: int,
        match_over: bool = False,
    ) -> ScoreHistoryEntry:
        entry = ScoreHistoryEntry(
            sequence=len(self._entries) + 1,
            game_number=int(game_number),
            event=str(event),
            action=str(action),
            reason=str(reason),
            score_a=score_state.score_a,
            score_b=score_state.score_b,
            serving_team=score_state.serving_team,
            server_number=score_state.server_number,
            server_court=score_state.server_court,
            games_a=int(games_a),
            games_b=int(games_b),
            game_over=score_state.game_over,
            match_over=bool(match_over),
        )
        self._entries.append(entry)
        return entry
