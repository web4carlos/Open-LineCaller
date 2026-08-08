import cv2
import numpy as np
import pytest

from linecaller.calibration.court_model import PickleballCourtModel
from linecaller.calibration.quality import CalibrationStatus
from linecaller.calibration.session import CalibrationSession


def synthetic_points():
    court = PickleballCourtModel()
    session = CalibrationSession(court=court)
    court_points = np.asarray(session.court_points(), dtype=np.float32)

    corners_court = np.asarray([
        court.reference_points()["near_left_corner"],
        court.reference_points()["near_right_corner"],
        court.reference_points()["far_right_corner"],
        court.reference_points()["far_left_corner"],
    ], dtype=np.float32)

    corners_image = np.asarray([
        (200.0, 850.0),
        (1100.0, 840.0),
        (880.0, 160.0),
        (410.0, 170.0),
    ], dtype=np.float32)

    court_to_image = cv2.getPerspectiveTransform(
        corners_court,
        corners_image,
    )

    image_points = cv2.perspectiveTransform(
        court_points.reshape(-1, 1, 2),
        court_to_image,
    ).reshape(-1, 2)

    return session, image_points


def test_session_requires_all_points():
    session = CalibrationSession()
    with pytest.raises(ValueError):
        session.build(name="x", image_size=(1280, 960))


def test_session_builds_valid_synthetic_calibration():
    session, image_points = synthetic_points()

    for x, y in image_points:
        session.add_point(float(x), float(y))

    result = session.build(
        name="synthetic",
        image_size=(1280, 960),
    )

    assert result.profile.status == CalibrationStatus.VALID
    assert result.quality.metrics.mean_error_px < 0.01


def test_session_undo_and_reset():
    session = CalibrationSession()
    session.add_point(10, 20)
    session.add_point(30, 40)

    session.undo()
    assert len(session.image_points) == 1

    session.reset()
    assert len(session.image_points) == 0


def test_overlay_contains_official_lines():
    session, image_points = synthetic_points()

    for x, y in image_points:
        session.add_point(float(x), float(y))

    session.build(name="synthetic", image_size=(1280, 960))

    names = {segment.name for segment in session.overlay_segments()}

    assert "near_baseline" in names
    assert "far_baseline" in names
    assert "left_sideline" in names
    assert "right_sideline" in names
    assert "near_nvz" in names
    assert "far_nvz" in names
