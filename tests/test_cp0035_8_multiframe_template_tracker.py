import cv2
import numpy as np
from pathlib import Path

from linecaller.dcf.multiframe_template_tracker import DCFMultiFrameTemplateTracker


def test_roi_is_centered_and_clamped():
    t=DCFMultiFrameTemplateTracker()
    roi=t._roi(10,10,30,20,(100,200,3))
    assert roi==(0,0,40,30)


def test_tracker_has_multiscale_matcher():
    t=DCFMultiFrameTemplateTracker()
    assert 0.5 in t.matcher.scales
    assert 2.0 in t.matcher.scales


def test_lost_roi_expansion_configured():
    t=DCFMultiFrameTemplateTracker(lost_expand_px=40,max_lost_frames=5)
    assert t.lost_expand_px==40
    assert t.max_lost_frames==5
