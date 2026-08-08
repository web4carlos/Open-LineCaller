# CP-0006 Technical Design

## Goal

Exercise the full officiating path on recorded video.

## Pipeline

1. VideoSource reads frames.
2. MotionBallDetector emits candidate balls.
3. BallEngine tracks the most plausible candidate.
4. Each tracked/predicted point is appended to trajectory.
5. BounceEngine confirms temporal bounce events.
6. DecisionEngine fuses each bounce with a validated calibration.
7. OverlayRenderer visualizes track/bounce/decision.
8. DecisionEventStore writes explainable JSONL records.
9. VideoWriter produces an annotated MP4.

## Detector baseline

The baseline detector combines:

- background subtraction
- contour area limits
- aspect ratio
- circularity
- short radius limits
- optional brightness preference

This is deliberately conservative and configurable.

## Safety

No calibration = no automatic decision.
Invalid calibration = REVIEW.
Low bounce confidence = REVIEW.
Uncertainty crossing a line = REVIEW.

## Performance

CP-0006 prioritizes correctness and observability over maximum FPS.
Later optimization can decouple detection, rendering, and encoding.
