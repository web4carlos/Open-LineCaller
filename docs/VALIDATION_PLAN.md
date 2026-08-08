# Calibration Validation Plan

For each supported camera setup:

1. Capture a still frame with the entire court visible.
2. Mark at least:
   - four outer corners
   - two NVZ/sideline intersections
   - two centerline intersections when visible
3. Compute homography.
4. Reproject all marked points.
5. Record:
   - mean error (px)
   - max error (px)
   - RMS error (px)
   - inlier count
6. Reject calibration when thresholds fail.
7. Repeat after:
   - camera repositioning
   - indoor/outdoor lighting change
   - lens zoom/focus change

Future validation must also measure physical error in millimeters near the
sidelines and baselines, because pixel error alone is not sufficient for final
line-call accuracy.
