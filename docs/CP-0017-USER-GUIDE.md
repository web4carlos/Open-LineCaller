# CP-0017 User Guide

Run:

    cd C:\olinecaller
    .\.venv\Scripts\python.exe -m tools.live_match_app

The live window now has a pipeline integration point.

When a valid pipeline event is produced:
- IN / OUT appears immediately
- audio hook is triggered
- REVIEW activates replay

If tracking is not reliable, the UI stays SEARCHING or LOST instead of making
a fake call.
