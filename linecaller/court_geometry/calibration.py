from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CourtCalibration:
    """
    Pixel point order:
      0 = near-left
      1 = near-right
      2 = far-right
      3 = far-left

    Court coordinates use feet:
      x: 0..20
      y: 0..44

    Mapping:
      near-left  -> (0, 44)
      near-right -> (20, 44)
      far-right  -> (20, 0)
      far-left   -> (0, 0)
    """
    image_points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    court_width_ft: float = 20.0
    court_length_ft: float = 44.0

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": 1,
                    "point_order": [
                        "near-left",
                        "near-right",
                        "far-right",
                        "far-left",
                    ],
                    "image_points": [
                        [float(x), float(y)]
                        for x, y in self.image_points
                    ],
                    "court_width_ft": self.court_width_ft,
                    "court_length_ft": self.court_length_ft,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        d = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        pts = d.get("image_points", [])

        if len(pts) != 4:
            raise ValueError(
                "Court calibration requires exactly 4 image_points."
            )

        return cls(
            image_points=tuple(
                (float(p[0]), float(p[1]))
                for p in pts
            ),
            court_width_ft=float(d.get("court_width_ft", 20.0)),
            court_length_ft=float(d.get("court_length_ft", 44.0)),
        )
