# CP-0015 User Guide

## Live smoke test

Connect a camera and run:

    cd C:\olinecaller
    .\.venv\Scripts\python.exe -m tools.live_camera_test --camera 0

The window shows:
- live FPS
- frames captured
- replay-buffer size

Press Q to exit.

## What REVIEW means

REVIEW means:

    SHOW INSTANT REPLAY

It is not a third line-call result.

## Accuracy target

Open-LineCaller is not considered ready for competitive live use until the
validation suite demonstrates >= 98% automatic IN/OUT accuracy under supported
conditions.
