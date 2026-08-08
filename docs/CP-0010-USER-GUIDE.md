# CP-0010 User Guide

## Auto Calibration

1. Open a video.
2. Pick a clear frame.
3. Click **AUTO CALIBRATE**.
4. Review:
   - disposition
   - confidence/score
   - score margin
   - number of hypotheses
5. The four outer corners will be pre-filled if a proposal exists.
6. Drag any outer corner that does not align exactly.
7. Add the four Kitchen corners manually.
8. Click **Compute Calibration**.
9. Only save/use the calibration if it is VALID.

## If result is REJECT

Use a clearer frame or manually place all points.

## If result is AUTO_REVIEW

Inspect the proposed outer corners carefully before continuing.

## If result is AUTO_ACCEPT

Still inspect the overlay. AUTO_ACCEPT is a prefill confidence, not final court
validation.
