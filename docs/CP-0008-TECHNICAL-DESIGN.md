# CP-0008 Technical Design — Assisted Auto-Calibration

## Goal

Reduce manual calibration effort while preserving the existing quality gate.

## Stages

### 1. Edge detection
Grayscale -> Gaussian blur -> Canny.

### 2. Hough lines
Probabilistic Hough transform returns finite line segments.

### 3. Orientation classification
Segments are normalized to an orientation angle in [0, 180).

For perspective court images, strict "horizontal/vertical" is too simplistic.
We instead classify into two dominant orientation families.

### 4. Dominant families
A histogram over orientation angle finds the strongest direction family.
The second family must be sufficiently separated in angle.

### 5. Candidate boundary lines
Long lines are ranked within each family and duplicates are merged by
normal-distance similarity.

### 6. Intersections
Cross-family line intersections are computed and filtered to image bounds
with a configurable margin.

### 7. Quadrilateral proposal
Four boundary candidates are selected to form a plausible convex court
quadrilateral.

### 8. Confidence
Evidence score combines:
- total line evidence
- two-family separation
- quadrilateral area
- convexity
- corner visibility
- line support

## Safety

The result is a proposal only.

`AUTO_ACCEPT` means:
"good enough to pre-fill points"

It does NOT mean:
"safe enough to officiate"

Final officiating still requires:
CalibrationProfile.status == VALID.
