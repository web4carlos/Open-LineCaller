from __future__ import annotations

import math

from .models import BallProposal


def center_distance(a: BallProposal, b: BallProposal) -> float:
    return math.hypot(
        a.center_x - b.center_x,
        a.center_y - b.center_y,
    )


def iou(a: BallProposal, b: BallProposal) -> float:
    ax2 = a.x + a.width
    ay2 = a.y + a.height
    bx2 = b.x + b.width
    by2 = b.y + b.height

    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0
