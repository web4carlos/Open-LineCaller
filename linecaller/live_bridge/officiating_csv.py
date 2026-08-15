from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CsvOfficiatingEvent:
    frame: int
    final_decision: str
    decision_confidence: float
    image_x: float
    image_y: float
    nearest_line: str
    signed_distance_in: float
    geometry_state: str
    bounce_confidence: float
    bounce_score: float
    gate_decision: str
    gate_confidence: float
    force_review: bool
    officiating_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CsvOfficiatingTimeline:
    """
    Adapter for the validated CP-0029.1 final officiating CSV.

    This is intentionally downstream of the existing pipeline:
      bounce -> geometry -> base decision -> officiating gate -> final decision.

    React receives only the final trusted/review result. It never recomputes it.
    """

    REQUIRED = {
        "frame",
        "image_x",
        "image_y",
        "nearest_line",
        "geometry_state",
        "bounce_confidence",
        "bounce_score",
        "decision_confidence",
        "signed_distance_in",
        "gate_decision",
        "gate_confidence",
        "force_review",
        "final_decision",
        "officiating_reason",
    }

    def __init__(self, events: tuple[CsvOfficiatingEvent, ...] = ()):
        self.events = tuple(sorted(events, key=lambda x: x.frame))

    @classmethod
    def load(cls, path: str | Path) -> "CsvOfficiatingTimeline":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)

        events: list[CsvOfficiatingEvent] = []
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or ())
            missing = cls.REQUIRED - fields
            if missing:
                raise ValueError(
                    "missing required officiating CSV fields: "
                    + ", ".join(sorted(missing))
                )

            for row_no, row in enumerate(reader, start=2):
                decision = str(row["final_decision"]).strip().upper()
                if decision not in {"IN", "OUT", "REVIEW"}:
                    raise ValueError(
                        f"invalid final_decision {decision!r} at row {row_no}"
                    )

                force_review = _bool(row["force_review"])
                # Integrity guard: forced review cannot publish a terminal call.
                if force_review and decision != "REVIEW":
                    raise ValueError(
                        f"force_review=True but final_decision={decision} "
                        f"at row {row_no}"
                    )

                events.append(
                    CsvOfficiatingEvent(
                        frame=int(row["frame"]),
                        final_decision=decision,
                        decision_confidence=float(row["decision_confidence"]),
                        image_x=float(row["image_x"]),
                        image_y=float(row["image_y"]),
                        nearest_line=str(row["nearest_line"]),
                        signed_distance_in=float(row["signed_distance_in"]),
                        geometry_state=str(row["geometry_state"]),
                        bounce_confidence=float(row["bounce_confidence"]),
                        bounce_score=float(row["bounce_score"]),
                        gate_decision=str(row["gate_decision"]),
                        gate_confidence=float(row["gate_confidence"]),
                        force_review=force_review,
                        officiating_reason=str(row["officiating_reason"]),
                    )
                )

        return cls(tuple(events))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
