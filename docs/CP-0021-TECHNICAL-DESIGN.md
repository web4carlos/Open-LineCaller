# CP-0021 Technical Design — Advanced Candidate Generation

BallEngine expects a detector with:

    detect(frame) -> list[BallCandidate]

AdvancedMotionBallDetector implements that contract directly and is incremental,
so it is compatible with LIVE camera frames.

Features:
1. MOG2 foreground segmentation.
2. Median blur and morphology.
3. Area/radius filtering.
4. Circularity/aspect scoring.
5. HSV brightness/saturation.
6. Broad yellow-green affinity.
7. Temporal proximity to previous best candidate.
8. Max candidates per frame.
9. Telemetry and reset.

This remains candidate generation only; Tracker, BounceEngine and DecisionEngine
retain their existing responsibilities.
