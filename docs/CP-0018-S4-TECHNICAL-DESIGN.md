# CP-0018 Sprint 4 Technical Design — Professional Live HUD

## Goal

Communicate live system state clearly without modifying the officiating engine.

## HUD semantics

StatusLevel:
- OK
- WARN
- ERROR
- UNKNOWN

Tracking display states:
- SEARCHING
- TRACK FOUND
- LOCKED
- LOST
- REACQUIRING

Latency quality:
- EXCELLENT
- ACCEPTABLE
- HIGH
- UNKNOWN

Confidence:
- numeric percentage
- visual progress bar
- derived status level

Last call:
- IN
- OUT
- REVIEW
- idle

## Separation of concerns

The HUD receives state and renders it.
It does not determine whether a call is correct.
