from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from linecaller.rules.models import TeamSide


class MatchFormat(str, Enum):
    SINGLES = "SINGLES"
    DOUBLES = "DOUBLES"


@dataclass(frozen=True)
class ScoreState:
    score_a: int = 0
    score_b: int = 0
    serving_team: TeamSide = TeamSide.A
    server_number: int = 2
    match_format: MatchFormat = MatchFormat.DOUBLES
    target_score: int = 11
    win_by: int = 2
    game_over: bool = False
    winner: TeamSide = TeamSide.UNKNOWN
    rallies_played: int = 0

    @property
    def serving_score(self) -> int:
        return self.score_a if self.serving_team == TeamSide.A else self.score_b

    @property
    def receiving_team(self) -> TeamSide:
        return TeamSide.B if self.serving_team == TeamSide.A else TeamSide.A

    @property
    def server_court(self) -> str:
        return "RIGHT" if self.serving_score % 2 == 0 else "LEFT"


@dataclass(frozen=True)
class ScoreUpdate:
    state: ScoreState
    action: str
    reason: str


class SideOutScoreEngine:
    """
    CP-0033.0 — Traditional Side-Out Score Engine.

    Doubles:
      - Only serving team scores.
      - Opening service sequence begins with server_number=2 so only one
        opening server is available before first side-out.
      - Thereafter each team receives server 1 then server 2.
      - Receiver rally win:
          server 1 -> server 2
          server 2 -> side-out

    Singles:
      - Only serving player scores.
      - Losing rally causes immediate side-out.
      - Right court on even serving score, left court on odd serving score.

    Game:
      - configurable target (default 11)
      - win by configurable margin (default 2)
    """

    def __init__(
        self,
        *,
        match_format: MatchFormat = MatchFormat.DOUBLES,
        starting_team: TeamSide = TeamSide.A,
        target_score: int = 11,
        win_by: int = 2,
    ):
        initial_server = 2 if match_format == MatchFormat.DOUBLES else 1

        self.state = ScoreState(
            serving_team=starting_team,
            server_number=initial_server,
            match_format=match_format,
            target_score=int(target_score),
            win_by=int(win_by),
        )

    @staticmethod
    def _other(team: TeamSide) -> TeamSide:
        return TeamSide.B if team == TeamSide.A else TeamSide.A

    @staticmethod
    def _score_for(state: ScoreState, team: TeamSide) -> int:
        return state.score_a if team == TeamSide.A else state.score_b

    def _check_game(self, state: ScoreState) -> ScoreState:
        a = state.score_a
        b = state.score_b

        if (
            a >= state.target_score
            and a - b >= state.win_by
        ):
            return replace(
                state,
                game_over=True,
                winner=TeamSide.A,
            )

        if (
            b >= state.target_score
            and b - a >= state.win_by
        ):
            return replace(
                state,
                game_over=True,
                winner=TeamSide.B,
            )

        return state

    def apply_rally_winner(self, winner: TeamSide) -> ScoreUpdate:
        s = self.state

        if s.game_over:
            return ScoreUpdate(
                s,
                "IGNORED",
                "GAME_ALREADY_OVER",
            )

        if winner not in (TeamSide.A, TeamSide.B):
            return ScoreUpdate(
                s,
                "IGNORED",
                "UNKNOWN_RALLY_WINNER",
            )

        # Serving side wins: point.
        if winner == s.serving_team:
            if winner == TeamSide.A:
                ns = replace(
                    s,
                    score_a=s.score_a + 1,
                    rallies_played=s.rallies_played + 1,
                )
            else:
                ns = replace(
                    s,
                    score_b=s.score_b + 1,
                    rallies_played=s.rallies_played + 1,
                )

            ns = self._check_game(ns)
            self.state = ns

            return ScoreUpdate(
                ns,
                "POINT",
                "SERVING_TEAM_WON_RALLY",
            )

        # Receiving side wins: no point in traditional side-out scoring.
        if s.match_format == MatchFormat.SINGLES:
            ns = replace(
                s,
                serving_team=winner,
                server_number=1,
                rallies_played=s.rallies_played + 1,
            )
            self.state = ns
            return ScoreUpdate(
                ns,
                "SIDE_OUT",
                "SINGLES_RECEIVER_WON_RALLY",
            )

        # Doubles: first server loses -> second server.
        if s.server_number == 1:
            ns = replace(
                s,
                server_number=2,
                rallies_played=s.rallies_played + 1,
            )
            self.state = ns
            return ScoreUpdate(
                ns,
                "SECOND_SERVER",
                "FIRST_SERVER_LOST_RALLY",
            )

        # Second server loses -> side out; new team starts with server 1.
        ns = replace(
            s,
            serving_team=winner,
            server_number=1,
            rallies_played=s.rallies_played + 1,
        )
        self.state = ns

        return ScoreUpdate(
            ns,
            "SIDE_OUT",
            "SECOND_SERVER_LOST_RALLY",
        )

    def score_call(self) -> str:
        """
        Traditional doubles score call from serving team's perspective:
        serving score - receiving score - server number.

        Singles returns:
        serving score - receiving score.
        """
        s = self.state
        serving = self._score_for(s, s.serving_team)
        receiving = self._score_for(s, s.receiving_team)

        if s.match_format == MatchFormat.DOUBLES:
            return f"{serving}-{receiving}-{s.server_number}"

        return f"{serving}-{receiving}"
