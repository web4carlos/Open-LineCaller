"""
Compatibility adapter.

CP-0035.9.1 exposed OneBallColorDCF, DCFCell and ColorMatch from this module.
CP-0035.9.2 preserves those imports while delegating behaviour to the strict
True DCF Cell Scanner.
"""

from .true_cell_scanner import (
    BallColorProfile,
    ColorMatch,
    DCFCell,
    TrueDCFCellScanner,
)


class OneBallColorDCF(TrueDCFCellScanner):
    """Backward-compatible name for the CP-0035.9.2 scanner."""
    pass


__all__ = [
    "BallColorProfile",
    "ColorMatch",
    "DCFCell",
    "OneBallColorDCF",
    "TrueDCFCellScanner",
]
