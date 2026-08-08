from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np

from linecaller.calibration.quality import CalibrationStatus


@dataclass
class CalibrationProfile:
    name: str
    image_size: tuple[int, int]
    image_points: list[tuple[float, float]]
    court_points: list[tuple[float, float]]
    homography: np.ndarray
    status: CalibrationStatus
    mean_error_px: float
    max_error_px: float
    rms_error_px: float
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        image_size: tuple[int, int],
        image_points: list[tuple[float, float]],
        court_points: list[tuple[float, float]],
        homography: np.ndarray,
        status: CalibrationStatus,
        mean_error_px: float,
        max_error_px: float,
        rms_error_px: float,
    ) -> "CalibrationProfile":
        return cls(
            name=name,
            image_size=image_size,
            image_points=image_points,
            court_points=court_points,
            homography=homography,
            status=status,
            mean_error_px=mean_error_px,
            max_error_px=max_error_px,
            rms_error_px=rms_error_px,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "image_size": list(self.image_size),
            "image_points": [list(p) for p in self.image_points],
            "court_points": [list(p) for p in self.court_points],
            "homography": self.homography.tolist(),
            "status": self.status.value,
            "mean_error_px": self.mean_error_px,
            "max_error_px": self.max_error_px,
            "rms_error_px": self.rms_error_px,
            "created_at": self.created_at,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            image_size=tuple(data["image_size"]),
            image_points=[tuple(p) for p in data["image_points"]],
            court_points=[tuple(p) for p in data["court_points"]],
            homography=np.asarray(data["homography"], dtype=np.float64),
            status=CalibrationStatus(data["status"]),
            mean_error_px=float(data["mean_error_px"]),
            max_error_px=float(data["max_error_px"]),
            rms_error_px=float(data["rms_error_px"]),
            created_at=data["created_at"],
        )
