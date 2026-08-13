from __future__ import annotations

from dataclasses import dataclass, replace

from linecaller.rules.models import TeamSide
from .engine import MatchFormat, SideOutScoreEngine
from .rally_bridge import RallyOutcomeScoreBridge


@dataclass(frozen=True)
class MatchState:
    games_a: int = 0
    games_b: int = 0
    current_game: int = 1
    best_of: int = 3
    match_over: bool = False
    match_winner: TeamSide = TeamSide.UNKNOWN
    starting_team: TeamSide = TeamSide.A


@dataclass(frozen=True)
class MatchUpdate:
    state: MatchState
    action: str
    reason: str


class MatchController:
    """
    CP-0033.2 — Match State + Game Lifecycle.

    Owns game-to-game lifecycle while SideOutScoreEngine owns points/service.
    A game is committed only after ScoreEngine marks game_over=True.
    """

    def __init__(
        self,
        *,
        best_of: int = 3,
        match_format: MatchFormat = MatchFormat.DOUBLES,
        starting_team: TeamSide = TeamSide.A,
        target_score: int = 11,
        win_by: int = 2,
        alternate_game_start: bool = True,
    ):
        if best_of < 1 or best_of % 2 == 0:
            raise ValueError("best_of must be a positive odd number")

        self.match_format = match_format
        self.target_score = int(target_score)
        self.win_by = int(win_by)
        self.alternate_game_start = bool(alternate_game_start)

        self.state = MatchState(
            best_of=best_of,
            starting_team=starting_team,
        )

        self.score_engine = self._new_score_engine(starting_team)
        self.bridge = RallyOutcomeScoreBridge(self.score_engine)

    @property
    def games_to_win(self) -> int:
        return self.state.best_of // 2 + 1

    def _new_score_engine(self, starting_team: TeamSide):
        return SideOutScoreEngine(
            match_format=self.match_format,
            starting_team=starting_team,
            target_score=self.target_score,
            win_by=self.win_by,
        )

    def _next_starting_team(self) -> TeamSide:
        if not self.alternate_game_start:
            return self.state.starting_team
        return (
            TeamSide.B
            if self.state.starting_team == TeamSide.A
            else TeamSide.A
        )

    def commit_finished_game(self) -> MatchUpdate:
        s = self.state
        game = self.score_engine.state

        if s.match_over:
            return MatchUpdate(s, "IGNORED", "MATCH_ALREADY_OVER")

        if not game.game_over or game.winner == TeamSide.UNKNOWN:
            return MatchUpdate(s, "IGNORED", "CURRENT_GAME_NOT_FINISHED")

        ga = s.games_a + (1 if game.winner == TeamSide.A else 0)
        gb = s.games_b + (1 if game.winner == TeamSide.B else 0)

        if ga >= self.games_to_win or gb >= self.games_to_win:
            winner = TeamSide.A if ga > gb else TeamSide.B
            self.state = replace(
                s,
                games_a=ga,
                games_b=gb,
                match_over=True,
                match_winner=winner,
            )
            return MatchUpdate(
                self.state,
                "MATCH_WON",
                f"MATCH_WON_BY_{winner.value}",
            )

        next_start = self._next_starting_team()

        self.state = replace(
            s,
            games_a=ga,
            games_b=gb,
            current_game=s.current_game + 1,
            starting_team=next_start,
        )

        self.score_engine = self._new_score_engine(next_start)
        self.bridge = RallyOutcomeScoreBridge(self.score_engine)

        return MatchUpdate(
            self.state,
            "NEW_GAME",
            f"GAME_WON_BY_{game.winner.value}",
        )

    def match_score(self) -> str:
        return f"{self.state.games_a}-{self.state.games_b}"
