# CP-0008 User Guide

## Recommended frame

Use a frame where:
- the full outer court is visible
- sidelines and baselines are not heavily occluded
- the Kitchen lines are visible if possible
- contrast between line paint and surface is good
- the camera is in the same fixed position used for the match

## What to expect

The assisted engine can propose the four outer corners automatically.

If confidence is high:
    AUTO_ACCEPT

If geometry is plausible but uncertain:
    AUTO_REVIEW

If the court cannot be isolated:
    REJECT

## What happens next

The proposed points will later be injected into Calibration Studio so you can
drag only the points that need correction.

That is the bridge from CP-0008 to CP-0009 full auto-calibration.
