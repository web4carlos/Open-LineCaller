from __future__ import annotations

from enum import Enum


class ProductRoute(str, Enum):
    HOME = "HOME"
    START_MATCH = "START_MATCH"
    HISTORY = "HISTORY"
    SETTINGS = "SETTINGS"
    DIAGNOSTICS = "DIAGNOSTICS"


class NavigationController:
    def __init__(self):
        self._current = ProductRoute.HOME
        self._history = [ProductRoute.HOME]

    @property
    def current(self) -> ProductRoute:
        return self._current

    @property
    def history(self) -> tuple[ProductRoute, ...]:
        return tuple(self._history)

    def navigate(self, route: ProductRoute | str) -> ProductRoute:
        route = ProductRoute(route)

        if route != self._current:
            self._current = route
            self._history.append(route)

        return self._current

    def home(self) -> ProductRoute:
        return self.navigate(ProductRoute.HOME)

    def back(self) -> ProductRoute:
        if len(self._history) <= 1:
            self._current = ProductRoute.HOME
            return self._current

        self._history.pop()
        self._current = self._history[-1]
        return self._current
