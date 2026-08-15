import cv2
import numpy as np

from linecaller.dcf.true_cell_scanner import DCFCell, TrueDCFCellScanner


def make_ball(color_bgr, size=17):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), max(3, size // 2 - 2), color_bgr, -1, cv2.LINE_AA)
    cv2.circle(img, (c - 2, c - 2), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def put(frame, patch, center):
    h, w = patch.shape[:2]
    x1 = int(round(center[0] - w / 2))
    y1 = int(round(center[1] - h / 2))
    frame[y1:y1 + h, x1:x1 + w] = patch


def cell():
    return DCFCell(
        ix=0,
        iy=0,
        x1=20,
        y1=20,
        x2=68,
        y2=68,
        inside=True,
        expected_scale=1.0,
    )


def test_true_cell_scanner_accepts_ball_at_projected_cell_center():
    ball = make_ball((0, 220, 255))  # yellow/orange
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    put(frame, ball, (44, 44))

    scanner = TrueDCFCellScanner(min_score=0.55)
    hit = scanner.search(frame, ball, [cell()])

    assert hit.found
    assert hit.cell == cell()
    assert abs(hit.x - 44) <= 2
    assert abs(hit.y - 44) <= 2
    assert hit.offset_px is not None
    assert hit.offset_px <= 2.0


def test_identical_ball_far_inside_legacy_48px_cell_is_rejected():
    """
    CP-0035.9.1 could accept this because it searched the whole 48px cell.
    CP-0035.9.2 must reject it because it is not at the projected cell centre.
    """
    ball = make_ball((0, 220, 255))
    frame = np.zeros((96, 96, 3), dtype=np.uint8)

    # Cell centre is (44,44). This is still inside the same 48px cell,
    # but deliberately too far from the physical hypothesis.
    put(frame, ball, (29, 44))

    scanner = TrueDCFCellScanner(min_score=0.55)
    hit = scanner.search(frame, ball, [cell()])

    assert not hit.found


def test_wrong_ball_color_is_rejected_even_at_correct_cell_center():
    yellow_ball = make_ball((0, 220, 255))
    blue_ball = make_ball((255, 80, 0))

    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    put(frame, blue_ball, (44, 44))

    scanner = TrueDCFCellScanner(
        min_score=0.50,
        min_color_score=0.60,
    )
    hit = scanner.search(frame, yellow_ball, [cell()])

    assert not hit.found


def test_search_cannot_silently_change_locked_ball_identity():
    yellow_ball = make_ball((0, 220, 255))
    blue_ball = make_ball((255, 80, 0))

    scanner = TrueDCFCellScanner(
        min_score=0.50,
        min_color_score=0.60,
    )

    first = np.zeros((96, 96, 3), dtype=np.uint8)
    put(first, yellow_ball, (44, 44))
    assert scanner.search(first, yellow_ball, [cell()]).found
    assert scanner.ball_generation == 1

    second = np.zeros((96, 96, 3), dtype=np.uint8)
    put(second, blue_ball, (44, 44))

    # Passing a different reference to search() must NOT replace the locked ball.
    assert not scanner.search(second, blue_ball, [cell()]).found
    assert scanner.ball_generation == 1


def test_replace_ball_event_changes_active_identity_without_touching_dcf():
    yellow_ball = make_ball((0, 220, 255))
    blue_ball = make_ball((255, 80, 0))

    scanner = TrueDCFCellScanner(
        min_score=0.50,
        min_color_score=0.60,
    )

    first = np.zeros((96, 96, 3), dtype=np.uint8)
    put(first, yellow_ball, (44, 44))
    assert scanner.search(first, yellow_ball, [cell()]).found

    scanner.replace_ball(blue_ball)
    assert scanner.ball_generation == 2

    second = np.zeros((96, 96, 3), dtype=np.uint8)
    put(second, blue_ball, (44, 44))
    hit = scanner.search(second, yellow_ball, [cell()])

    assert hit.found
    assert hit.ball_generation == 2
    assert hit.cell == cell()
