# Open-LineCaller CP-0002 — Precision Court Calibration

This capability pack adds a measurable calibration core for pickleball.

## What is included

- Official pickleball court geometry (20 ft x 44 ft, 7 ft NVZ)
- Homography calculation
- Image↔court coordinate transforms
- Reprojection error measurement
- Calibration quality gate: VALID / INVALID
- Save/load calibration profiles
- Lens calibration model container and undistortion helper
- Initial assisted line detector using OpenCV Hough lines
- Synthetic validation demo
- Pytest test suite
- Windows setup/test scripts (PowerShell + BAT)

## Important

This pack does **not** claim automatic production-grade court detection yet.
The assisted line detector proposes candidates only. Final calibration must be
validated by reprojection error before Open-LineCaller is allowed to make calls.

## Quick start (Windows)

If PowerShell blocks scripts, use the BAT files:

    setup.bat
    test.bat

Or:

    PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
    PowerShell -ExecutionPolicy Bypass -File .\test.ps1

Run the synthetic demo:

    .\.venv\Scripts\python.exe -m tools.calibration_demo

Expected:

    Calibration status: VALID
    Mean reprojection error: near 0 px

## Quality gate

The default gate is intentionally strict for synthetic/reference-point tests:

- mean reprojection error <= 3.0 px
- max reprojection error <= 6.0 px
- at least 6 reference points

These values are configurable and will be re-tuned later using real court footage.
