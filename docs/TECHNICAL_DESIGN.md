# CP-0002 Technical Design — Precision Court Calibration

## Purpose

The calibration subsystem establishes a mathematically measurable mapping between
camera image coordinates and official pickleball-court coordinates.

A simple four-corner homography can look visually acceptable and still be poor
near a line. Therefore CP-0002 introduces a quality gate based on multiple
reference points and reprojection error.

## Coordinate system

Court coordinates are measured in meters.

- origin: near-left outer corner
- +X: across court width
- +Y: toward far baseline
- court width: 6.096 m (20 ft)
- court length: 13.4112 m (44 ft)
- NVZ depth from net: 2.1336 m (7 ft)

## Pipeline

1. Optional lens-undistortion model
2. Collect image-space reference points
3. Match them with known court-space reference points
4. Estimate homography with RANSAC
5. Reproject all reference points
6. Compute mean/max/RMS error
7. Apply quality thresholds
8. Save calibration profile only as VALID or INVALID

## Safety rule

The officiating engine must not make automatic IN/OUT calls when calibration is
INVALID.

## Assisted line detection

The Hough-based detector in this pack is a candidate generator, not a final
court recognizer. A later pack will classify baseline/sideline/NVZ/centerline
candidates and feed them into robust court-model fitting.
