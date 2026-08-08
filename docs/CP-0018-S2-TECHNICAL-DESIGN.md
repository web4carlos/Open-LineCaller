# CP-0018 Sprint 2 Technical Design

## Goal

Introduce a deterministic pre-match state machine.

## Wizard states

CAMERA
CALIBRATION
HEALTH
READY

Calibration mode states:
AUTO
ASSISTED
MANUAL

## Rules

- camera must be selected before calibration
- auto calibration is preferred
- assisted is offered on failure
- manual remains final fallback
- READY cannot be reached until health check passes

## Integration

ProductAppWindow replaces START_MATCH placeholder with MatchWizardScreen.

Sprint 3 will make Health Check real and dynamic.
