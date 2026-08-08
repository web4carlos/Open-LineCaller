from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayMapping:
    frame_width: int
    frame_height: int
    widget_width: int
    widget_height: int

    @property
    def scale(self) -> float:
        return min(
            self.widget_width / self.frame_width,
            self.widget_height / self.frame_height,
        )

    @property
    def displayed_width(self) -> float:
        return self.frame_width * self.scale

    @property
    def displayed_height(self) -> float:
        return self.frame_height * self.scale

    @property
    def offset_x(self) -> float:
        return (self.widget_width - self.displayed_width) / 2.0

    @property
    def offset_y(self) -> float:
        return (self.widget_height - self.displayed_height) / 2.0

    def widget_to_frame(self, x: float, y: float) -> tuple[float, float] | None:
        local_x = x - self.offset_x
        local_y = y - self.offset_y

        if (
            local_x < 0
            or local_y < 0
            or local_x >= self.displayed_width
            or local_y >= self.displayed_height
        ):
            return None

        return (
            local_x / self.scale,
            local_y / self.scale,
        )
