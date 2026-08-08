from __future__ import annotations

import time
from typing import Any

import numpy as np

from .confidence import status_from_confidence
from .engine import ProposalEngine
from .models import (
    BallProposal,
    ProposalResult,
    ProposalSource,
)


class YOLOProposalEngine(ProposalEngine):
    """
    Ultralytics-backed ProposalEngine.

    Backend injection is supported so tests never need a real model.
    """

    def __init__(
        self,
        *,
        weights: str = "yolo11n.pt",
        class_name: str | None = "sports ball",
        min_confidence: float = 0.15,
        max_proposals: int = 10,
        backend: Any | None = None,
        fail_open: bool = True,
    ):
        self.weights = str(weights)
        self.class_name = class_name
        self.min_confidence = float(min_confidence)
        self.max_proposals = int(max_proposals)
        self._backend = backend
        self.fail_open = bool(fail_open)
        self.last_error: str | None = None

    def _load_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            from ultralytics import YOLO
            self._backend = YOLO(self.weights)
            self.last_error = None
            return self._backend
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

            if self.fail_open:
                return None

            raise RuntimeError(
                f"Unable to load YOLO model {self.weights}: {exc}"
            ) from exc

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return np.asarray([])

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "numpy"):
            value = value.numpy()

        return np.asarray(value)

    @staticmethod
    def _resolve_names(result, backend):
        names = getattr(result, "names", None)

        if names is None:
            names = getattr(backend, "names", None)

        return names or {}

    @staticmethod
    def _class_name(names, class_id: int) -> str | None:
        if isinstance(names, dict):
            value = names.get(class_id)
            return str(value) if value is not None else None

        if isinstance(names, (list, tuple)):
            if 0 <= class_id < len(names):
                return str(names[class_id])

        return None

    def propose(
        self,
        frame_number: int,
        frame: np.ndarray,
    ) -> ProposalResult:
        start = time.perf_counter()
        backend = self._load_backend()

        if backend is None:
            latency_ms = (time.perf_counter() - start) * 1000.0

            return ProposalResult(
                frame_number=frame_number,
                proposals=(),
                engine_name="YOLOProposalEngine(unavailable)",
                latency_ms=latency_ms,
            )

        try:
            results = backend.predict(
                source=frame,
                conf=self.min_confidence,
                verbose=False,
            )

            result = results[0] if results else None

            if result is None or getattr(result, "boxes", None) is None:
                proposals = []
            else:
                boxes = result.boxes
                xyxy = self._to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
                conf = self._to_numpy(getattr(boxes, "conf", None)).reshape(-1)
                cls = self._to_numpy(getattr(boxes, "cls", None)).reshape(-1)

                names = self._resolve_names(result, backend)

                proposals = []

                count = min(len(xyxy), len(conf), len(cls))

                for i in range(count):
                    class_id = int(cls[i])
                    detected_name = self._class_name(names, class_id)

                    if (
                        self.class_name is not None
                        and detected_name != self.class_name
                    ):
                        continue

                    confidence = float(conf[i])

                    if confidence < self.min_confidence:
                        continue

                    x1, y1, x2, y2 = map(float, xyxy[i])

                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        continue

                    proposals.append(
                        BallProposal(
                            frame_number=int(frame_number),
                            x=x1,
                            y=y1,
                            width=width,
                            height=height,
                            confidence=max(0.0, min(1.0, confidence)),
                            source=ProposalSource.YOLO,
                            status=status_from_confidence(confidence),
                            metadata={
                                "class_id": class_id,
                                "class_name": detected_name,
                                "weights": self.weights,
                            },
                        )
                    )

                proposals.sort(
                    key=lambda p: p.confidence,
                    reverse=True,
                )

                proposals = proposals[: self.max_proposals]

            latency_ms = (time.perf_counter() - start) * 1000.0
            self.last_error = None

            return ProposalResult(
                frame_number=frame_number,
                proposals=tuple(proposals),
                engine_name="YOLOProposalEngine",
                latency_ms=latency_ms,
            )

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

            if not self.fail_open:
                raise

            latency_ms = (time.perf_counter() - start) * 1000.0

            return ProposalResult(
                frame_number=frame_number,
                proposals=(),
                engine_name="YOLOProposalEngine(error)",
                latency_ms=latency_ms,
            )
