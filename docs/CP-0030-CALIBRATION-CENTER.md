# CP-0030 — Calibration Center

Product order:

1. AUTO CALIBRATION — primary/default
2. MANUAL CALIBRATION — fallback

AUTO:
- detects court-like lines
- estimates four outer court corners
- generates preview
- reports confidence
- A = accept
- R = retry on later frame
- M = manual fallback

MANUAL:
- four clicks
- same validated CP-0028 order:
  1 near-left
  2 near-right
  3 far-right
  4 far-left

Output:
court.json compatible with CP-0028 geometry.

CP-0031 will integrate this calibration into the Live MVP.
