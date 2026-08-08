# CP-0007 Technical Design — Calibration Studio

## Purpose

Generate a valid, inspectable, persistent `CalibrationProfile` without manually
editing JSON.

## Separation of responsibilities

`CalibrationSession`
- owns reference-point order
- stores selected image points
- builds CP-0002 calibration
- exposes projected court overlay lines

`CalibrationStudioWindow`
- video navigation
- click-to-image coordinate mapping
- user guidance
- save/load interaction
- overlay rendering

The UI does not implement homography math itself.

## Coordinate mapping

The displayed video uses aspect-ratio-preserving scaling. Mouse coordinates are
converted back into original source-frame pixel coordinates before they are
stored.

This is critical: storing QLabel coordinates would corrupt calibration whenever
the window is resized.

## Court overlay

After calibration, the official `PickleballCourtModel.line_segments()` are
projected from court coordinates back into image coordinates with the inverse
homography.

This gives immediate visual feedback for:
- baselines
- sidelines
- NVZ lines
- centerlines

## Quality gate

The existing CP-0002 `CalibrationService` remains the authority for
VALID / INVALID.

CP-0007 does not bypass that gate.

## Limitations

- manual selection is still required
- lens calibration is not yet interactive in this studio
- reprojection is based on selected fit points
- automatic court recognition comes next
