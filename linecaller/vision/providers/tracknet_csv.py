from __future__ import annotations

import csv
from pathlib import Path

from linecaller.vision.models import VisionDetection


class TrackNetCsvVisionProvider:
    """
    Offline evaluation provider backed by TrackNet's native prediction CSV:

        Frame,Visibility,X,Y

    Coordinates are expected in original video pixel space.
    """

    def __init__(self, *, csv_path):
        self.csv_path = Path(csv_path)
        self._rows = {}
        self._initialized = False

    @property
    def name(self):
        return "tracknet"

    def initialize(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"TrackNet prediction CSV not found: {self.csv_path}"
            )

        rows = {}

        with self.csv_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)

            required = {"Frame", "Visibility", "X", "Y"}
            actual = set(reader.fieldnames or [])

            if not required.issubset(actual):
                raise ValueError(
                    "TrackNet CSV must contain: "
                    "Frame,Visibility,X,Y"
                )

            for row in reader:
                frame = int(row["Frame"])
                visible = int(float(row["Visibility"]))

                if visible != 1:
                    rows[frame] = None
                    continue

                x = float(row["X"])
                y = float(row["Y"])

                if x < 0 or y < 0:
                    rows[frame] = None
                    continue

                rows[frame] = (x, y)

        self._rows = rows
        self._initialized = True

    def detect(
        self,
        *,
        frame,
        frame_number,
        timestamp,
    ):
        if not self._initialized:
            self.initialize()

        position = self._rows.get(int(frame_number))

        if position is None:
            return VisionDetection(
                frame_number=int(frame_number),
                x=None,
                y=None,
                confidence=0.0,
                status="LOST",
                source=self.name,
                metadata={
                    "timestamp": float(timestamp),
                    "csv": str(self.csv_path),
                },
            )

        x, y = position

        return VisionDetection(
            frame_number=int(frame_number),
            x=float(x),
            y=float(y),
            confidence=1.0,
            status="TRACKING",
            source=self.name,
            metadata={
                "timestamp": float(timestamp),
                "csv": str(self.csv_path),
            },
        )

    def reset(self):
        # Predictions are frame-addressed and have no temporal state.
        pass

    def shutdown(self):
        self._initialized = False
