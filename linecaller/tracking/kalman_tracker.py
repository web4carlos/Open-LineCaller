from __future__ import annotations
import cv2
import numpy as np

from .models import DetectionPoint, TrackPoint

class BallKalmanTracker:
    """
    Constant-velocity Kalman tracker.

    Detection present:
        predict -> correct with YOLO center

    Detection missing:
        predict only for a short configurable gap

    Important:
    raw coordinates are preserved separately from tracked/predicted coordinates.
    Bounce logic must prefer raw measurements whenever possible.
    """

    def __init__(
        self,
        *,
        max_gap: int = 4,
        min_detection_confidence: float = 0.05,
    ):
        self.max_gap = int(max_gap)
        self.min_detection_confidence = float(min_detection_confidence)
        self.missed_frames = 0
        self.initialized = False

        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.35
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def reset(self):
        self.missed_frames = 0
        self.initialized = False

    def _initialize(self, x: float, y: float):
        self.kf.statePost = np.array(
            [[x], [y], [0], [0]],
            dtype=np.float32,
        )
        self.kf.statePre = self.kf.statePost.copy()
        self.initialized = True
        self.missed_frames = 0

    def update(
        self,
        frame: int,
        detection: DetectionPoint | None,
    ) -> TrackPoint:
        valid = (
            detection is not None
            and detection.confidence >= self.min_detection_confidence
        )

        if valid and not self.initialized:
            self._initialize(detection.x, detection.y)
            return TrackPoint(
                frame=frame,
                raw_x=detection.x,
                raw_y=detection.y,
                tracked_x=detection.x,
                tracked_y=detection.y,
                confidence=detection.confidence,
                source="YOLO",
                missed_frames=0,
            )

        if not self.initialized:
            return TrackPoint(
                frame=frame,
                raw_x=None,
                raw_y=None,
                tracked_x=None,
                tracked_y=None,
                confidence=0.0,
                source="LOST",
                missed_frames=0,
            )

        prediction = self.kf.predict()
        px = float(prediction[0, 0])
        py = float(prediction[1, 0])

        if valid:
            measurement = np.array(
                [[detection.x], [detection.y]],
                dtype=np.float32,
            )
            corrected = self.kf.correct(measurement)

            self.missed_frames = 0

            return TrackPoint(
                frame=frame,
                raw_x=detection.x,
                raw_y=detection.y,
                tracked_x=float(corrected[0, 0]),
                tracked_y=float(corrected[1, 0]),
                confidence=detection.confidence,
                source="YOLO+KALMAN",
                missed_frames=0,
            )

        self.missed_frames += 1

        if self.missed_frames <= self.max_gap:
            confidence = max(
                0.05,
                0.55 * (
                    1.0
                    - (self.missed_frames - 1) / max(1, self.max_gap)
                ),
            )

            return TrackPoint(
                frame=frame,
                raw_x=None,
                raw_y=None,
                tracked_x=px,
                tracked_y=py,
                confidence=confidence,
                source="KALMAN-PREDICT",
                missed_frames=self.missed_frames,
            )

        self.reset()

        return TrackPoint(
            frame=frame,
            raw_x=None,
            raw_y=None,
            tracked_x=None,
            tracked_y=None,
            confidence=0.0,
            source="LOST",
            missed_frames=self.missed_frames,
        )
