# CP-0025 — Ball Tracker

Goal:
Turn sparse YOLO26 detections into a short-gap continuous trajectory.

Inputs:
- YOLO center x/y
- confidence
- frame number

Outputs:
- raw_x/raw_y
- tracked_x/tracked_y
- confidence
- source
- missed_frames

Sources:
- YOLO
- YOLO+KALMAN
- KALMAN-PREDICT
- LOST

Important design rule:
Bounce detection must retain access to raw measurements. Smoothed/predicted
coordinates are useful for continuity and visualization but must not silently
replace the actual observed bounce coordinates.

Default gap:
4 frames.

This pack does NOT implement bounce detection or IN/OUT.
