# Calibration Studio User Guide

## Choose a good frame

Prefer a frame where:

- all four outer court corners are visible
- both NVZ lines are visible
- players are not covering important line intersections
- motion blur is low
- the camera has not moved from the match position

## Clicking order

Follow the instruction shown above the video.

Click the exact intersection of painted line centerlines, not the outside edge
of the paint.

If you make a mistake, use **Undo Point** or **Reset Points**.

## Calibration result

Green/VALID means the numerical gate passed.

INVALID means do not use that calibration for automatic calls.

Even a VALID calibration should be visually inspected: the projected court
overlay should align with the real painted court over the entire playing area.

## Save

Use **Save Calibration...** and save a name such as:

    court_01.json

That JSON can be passed directly to CP-0006.
