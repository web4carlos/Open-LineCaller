# CP-0005 Validation Plan

Synthetic tests validate:

- valid inside point -> IN
- clear outside point -> OUT
- boundary-overlap uncertainty -> REVIEW
- invalid calibration -> REVIEW
- low bounce confidence -> REVIEW
- nearest-line identification
- signed distance convention
- local pixel-to-meter scale

Real-court validation must later measure physical error using known markers
placed at known distances from sidelines and baselines.

Required production metrics:

- median court-coordinate error (mm)
- 95th percentile court-coordinate error (mm)
- false IN rate near boundary
- false OUT rate near boundary
- REVIEW rate within uncertainty band
