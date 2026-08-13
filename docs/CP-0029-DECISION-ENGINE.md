# CP-0029 — Decision Engine

Inputs:
- CP-0026.2 bounce confidence / bounce score
- CP-0028 signed distance to nearest court boundary

Outputs:
- IN
- OUT
- REVIEW

Policy is intentionally conservative.

Line contact:
A ball whose contact footprint reaches the boundary line is treated as IN.

REVIEW exists so the MVP does not fabricate certainty near a line or when
bounce evidence is weak.

CP-0030 will provide Calibration Center:
AUTO CALIBRATION first, MANUAL CALIBRATION fallback.

CP-0031 will integrate calibration + live tracking + bounce + geometry +
decision into the Live MVP.
