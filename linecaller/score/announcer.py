from __future__ import annotations

from dataclasses import dataclass

from linecaller.rules.models import TeamSide
from .engine import MatchFormat, ScoreState


@dataclass(frozen=True)
class AnnouncerState:
    game_number: int
    games_a: int
    games_b: int
    score_a: int
    score_b: int
    serving_team: TeamSide
    server_number: int
    server_court: str
    score_call: str
    last_action: str
    last_reason: str
    game_over: bool
    match_over: bool
    match_winner: TeamSide

    def spoken_score(self) -> str:
        """
        Machine-friendly text for a future TTS layer.
        Does not perform audio output.
        """
        if self.match_over:
            return (
                f"Match complete. Team {self.match_winner.value} wins. "
                f"Games {self.games_a} to {self.games_b}."
            )

        if self.game_over:
            return (
                f"Game complete. Score {self.score_a} to {self.score_b}."
            )

        if self.server_number in (1, 2):
            return (
                f"{self.score_call}. "
                f"Team {self.serving_team.value} serving from "
                f"{self.server_court.lower()} court."
            )

        return self.score_call


def build_announcer_state(
    *,
    score_state: ScoreState,
    game_number: int,
    games_a: int,
    games_b: int,
    last_action: str = "START",
    last_reason: str = "",
    match_over: bool = False,
    match_winner: TeamSide = TeamSide.UNKNOWN,
) -> AnnouncerState:
    receiving_team = (
        TeamSide.B if score_state.serving_team == TeamSide.A else TeamSide.A
    )
    serving_score = (
        score_state.score_a
        if score_state.serving_team == TeamSide.A
        else score_state.score_b
    )
    receiving_score = (
        score_state.score_b
        if receiving_team == TeamSide.B
        else score_state.score_a
    )

    if score_state.match_format == MatchFormat.DOUBLES:
        call = f"{serving_score}-{receiving_score}-{score_state.server_number}"
    else:
        call = f"{serving_score}-{receiving_score}"

    return AnnouncerState(
        game_number=int(game_number),
        games_a=int(games_a),
        games_b=int(games_b),
        score_a=score_state.score_a,
        score_b=score_state.score_b,
        serving_team=score_state.serving_team,
        server_number=score_state.server_number,
        server_court=score_state.server_court,
        score_call=call,
        last_action=str(last_action),
        last_reason=str(last_reason),
        game_over=score_state.game_over,
        match_over=bool(match_over),
        match_winner=match_winner,
    )
