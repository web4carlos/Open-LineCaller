from __future__ import annotations

from dataclasses import dataclass

from .models import TimedPoint3D
from .rolling_z0 import RollingThreePointZ0Predictor, RollingZ0Prediction


@dataclass(frozen=True)
class HitAnchoredZ0State:
    """State of one flight epoch anchored at the paddle-contact observation.

    The engine does not build a complete flight curve.  It keeps at most the
    latest three trusted same-ball observations.  The first rolling window is
    HIT + P1 + P2; each later observation drops the oldest point and produces
    a fresh Z=0 estimate.
    """

    epoch: int
    hit_anchor: TimedPoint3D | None
    points: tuple[TimedPoint3D, ...]
    prediction: RollingZ0Prediction | None

    @property
    def ready(self) -> bool:
        return self.prediction is not None


class HitAnchoredRollingZ0:
    """Paddle hit -> rolling 3-point estimates -> next Z=0.

    Contract:
      * ``start_epoch(hit)`` is called once a direction-changing paddle impact
        has been established.
      * The hit observation is point #1 of the first three-point window.
      * Two trusted post-hit observations produce the first Z=0 estimate.
      * Every later point shifts the window by one: P1,P2,P3 -> P2,P3,P4.
      * A new paddle hit resets the old window and starts a new epoch.
      * No full path is created or retained.
    """

    def __init__(self, *, gravity_bu_s2: float | None = None) -> None:
        self.predictor = RollingThreePointZ0Predictor(gravity_bu_s2=gravity_bu_s2)
        self._epoch = 0
        self._hit: TimedPoint3D | None = None
        self._points: list[TimedPoint3D] = []
        self._prediction: RollingZ0Prediction | None = None

    @property
    def state(self) -> HitAnchoredZ0State:
        return HitAnchoredZ0State(
            epoch=self._epoch,
            hit_anchor=self._hit,
            points=tuple(self._points),
            prediction=self._prediction,
        )

    def start_epoch(self, hit_observation: TimedPoint3D) -> HitAnchoredZ0State:
        self._epoch += 1
        self._hit = hit_observation
        self._points = [hit_observation]
        self._prediction = None
        return self.state

    def add(self, observation: TimedPoint3D) -> RollingZ0Prediction | None:
        if self._hit is None:
            raise RuntimeError("start_epoch(hit_observation) must be called first")
        if observation.t_s <= self._points[-1].t_s:
            raise ValueError("new observation time must be greater than previous time")

        self._points.append(observation)
        if len(self._points) > 3:
            self._points = self._points[-3:]

        if len(self._points) < 3:
            self._prediction = None
            return None

        self._prediction = self.predictor.predict(tuple(self._points))
        return self._prediction

    def reset(self) -> None:
        self._hit = None
        self._points = []
        self._prediction = None
