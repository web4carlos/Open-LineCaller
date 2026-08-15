import cv2
import numpy as np

from linecaller.dcf.one_ball_color_field import OneBallColorDCF, DCFCell


def test_color_prefers_yellow_ball_over_gray_distractor():
    frame = np.zeros((120, 220, 3), dtype=np.uint8)

    # Real target: yellow/green ball.
    cv2.circle(frame, (60, 60), 7, (0, 255, 255), -1)

    # Distractor with similar brightness/shape but gray.
    cv2.circle(frame, (160, 60), 7, (220, 220, 220), -1)

    template = frame[51:69, 51:69].copy()

    cells = [
        DCFCell(0, 0, 20, 30, 100, 90, True, 1.0),
        DCFCell(1, 0, 120, 30, 200, 90, False, 1.0),
    ]

    hit = OneBallColorDCF(.45).search(frame, template, cells)

    assert hit.found
    assert hit.cell.ix == 0
    assert abs(hit.x - 60) < 2


def test_expected_scale_is_used():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 50), 6, (0, 255, 255), -1)
    template = frame[43:57, 93:107].copy()

    cells = [DCFCell(0, 0, 70, 20, 140, 80, True, 1.0)]
    hit = OneBallColorDCF(.45).search(frame, template, cells)

    assert hit.found
