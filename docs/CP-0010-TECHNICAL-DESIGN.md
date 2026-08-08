# CP-0010 Technical Design — Smart Calibration Studio

## Goal

Turn assisted auto-calibration into a usable desktop workflow.

## Integration

CalibrationStudioWindow
  -> MultiHypothesisAutoCalibrationEngine
  -> RankedAutoCalibration
  -> best court quadrilateral
  -> SmartCalibrationAdapter
  -> outer-corner prefill
  -> manual refinement
  -> existing CalibrationSession
  -> existing CalibrationService
  -> VALID / INVALID

## Point mapping

CP-0009 returns outer corners in this order:

1. bottom-left
2. bottom-right
3. top-right
4. top-left

These map directly to:

- near_left_corner
- near_right_corner
- far_right_corner
- far_left_corner

Kitchen points are intentionally NOT guessed from the outer quadrilateral in
CP-0010. Guessing them geometrically from a single perspective quadrilateral
would create false precision.

The user still places/refines the four Kitchen intersections until the
full 8-point calibration is complete.

## Why not infer Kitchen automatically yet?

Because:
- perspective can vary strongly
- painted court markings may not follow simple screen-space interpolation
- independent line evidence is preferable

Future CP-0011/CP-0012 will detect and classify internal court lines explicitly.
