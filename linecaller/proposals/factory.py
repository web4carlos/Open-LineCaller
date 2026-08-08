from __future__ import annotations

import os

from .yolo_engine import YOLOProposalEngine


def create_default_proposal_engine():
    weights = os.environ.get(
        "OPEN_LINECALLER_YOLO_WEIGHTS",
        "yolo11n.pt",
    )

    class_name = os.environ.get(
        "OPEN_LINECALLER_YOLO_CLASS",
        "sports ball",
    )

    if class_name.strip().lower() in ("", "none", "*"):
        class_name = None

    return YOLOProposalEngine(
        weights=weights,
        class_name=class_name,
        min_confidence=0.15,
        max_proposals=10,
        fail_open=True,
    )
