# CP-0005 Technical Design — Court Fusion + Decision Geometry

## Goal

Transform a bounce event from image coordinates into a measurable, explainable
court decision.

## Pipeline

BounceEvent(image x,y)
    -> validated CalibrationProfile
    -> homography image->court
    -> court point (meters)
    -> nearest legal outer boundary
    -> signed distance
    -> local uncertainty estimate
    -> DecisionContext
    -> DecisionResult(IN / OUT / REVIEW)

## Signed distance

For the legal outer rectangle:

- positive = point is inside the legal court
- zero = exactly on the legal boundary
- negative = point is outside

## Nearest line

The engine considers the four rally boundaries:

- near baseline
- far baseline
- left sideline
- right sideline

NVZ and center lines are not rally out-of-bounds lines.

## Uncertainty propagation

CP-0002 currently reports reprojection error in pixels.

CP-0005 estimates local meters-per-pixel by projecting small image offsets
through the homography near the bounce point.

Approximate total image uncertainty:

    sqrt(calibration_error_px^2 + bounce_localization_error_px^2)

This is converted into meters using local image scale.

## Decision policy

Let:
- d = signed boundary distance in meters
- u = total geometric uncertainty in meters

Then:

- if evidence invalid -> REVIEW
- if |d| <= u -> REVIEW
- if d > u -> IN
- if d < -u -> OUT

This prevents false precision around close calls.

## Future refinement

The final officiating engine still needs:
- actual ball-floor contact-point estimation
- ball footprint / deformation model
- physical millimeter validation on real courts
- camera pose / lens validation under supported hardware setups
