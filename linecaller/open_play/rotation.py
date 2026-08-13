from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RotationPolicy(str, Enum):
    FOUR_ON_FOUR_OFF="FOUR_ON_FOUR_OFF"


@dataclass(frozen=True)
class OpenPlayPlayer:
    player_id: str
    name: str


@dataclass(frozen=True)
class RotationResult:
    action: str
    reason: str
    on_court: tuple[OpenPlayPlayer,...]
    waiting: tuple[OpenPlayPlayer,...]
    rotated_off: tuple[OpenPlayPlayer,...]=()


class OpenPlayRotationEngine:
    """
    CP-0033.5 — Open Play Rotation Engine.

    FOUR_ON_FOUR_OFF:
    - exactly four players form the active doubles group;
    - after a completed game, all four active players rotate off;
    - if four or more players are waiting, the next four enter;
    - the departing four join the back of the queue;
    - if fewer than four are waiting, the court waits rather than creating
      an incomplete doubles game.
    """

    COURT_SIZE=4

    def __init__(self, *, policy=RotationPolicy.FOUR_ON_FOUR_OFF):
        self.policy=policy
        self._waiting:list[OpenPlayPlayer]=[]
        self._on_court:list[OpenPlayPlayer]=[]

    @property
    def waiting(self): return tuple(self._waiting)

    @property
    def on_court(self): return tuple(self._on_court)

    def add_player(self, player:OpenPlayPlayer):
        ids={p.player_id for p in self._waiting+self._on_court}
        if player.player_id in ids:
            return RotationResult("NO_CHANGE","PLAYER_ALREADY_PRESENT",self.on_court,self.waiting)
        self._waiting.append(player)
        return RotationResult("QUEUED","PLAYER_ADDED_TO_QUEUE",self.on_court,self.waiting)

    def fill_court(self):
        if self._on_court:
            return RotationResult("NO_CHANGE","COURT_ALREADY_OCCUPIED",self.on_court,self.waiting)
        if len(self._waiting)<self.COURT_SIZE:
            return RotationResult("WAITING","NOT_ENOUGH_PLAYERS_FOR_DOUBLES",self.on_court,self.waiting)
        self._on_court=self._waiting[:4]
        del self._waiting[:4]
        return RotationResult("FOUR_ON","NEXT_FOUR_ENTERED_COURT",self.on_court,self.waiting)

    def complete_game(self):
        if len(self._on_court)!=4:
            return RotationResult("NO_CHANGE","NO_COMPLETE_ACTIVE_FOURSOME",self.on_court,self.waiting)

        departing=list(self._on_court)

        # Fair 4-on/4-off requires a complete next foursome.
        if len(self._waiting)<4:
            return RotationResult(
                "WAITING","FEWER_THAN_FOUR_WAITING_NO_ROTATION",
                self.on_court,self.waiting
            )

        next_four=self._waiting[:4]
        del self._waiting[:4]
        self._waiting.extend(departing)
        self._on_court=next_four

        return RotationResult(
            "FOUR_OFF_FOUR_ON",
            "GAME_COMPLETE_ALL_FOUR_ROTATED",
            self.on_court,self.waiting,tuple(departing)
        )
