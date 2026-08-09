# CP-0018 Sprint 3 Technical Design — Real Health Check

## Goal

Replace the Match Wizard's fake health check with deterministic subsystem checks.

## Components

HealthStatus
- PASS
- WARN
- FAIL

HealthCheckItem
- name
- status
- message
- recovery

HealthReport
- list of items
- ready property

HealthCheckService
- camera check
- calibration check
- replay check
- audio check
- tracking check
- fps check
- latency check
- storage check

## Policy

READY requires all required checks to PASS.

WARN does not block READY unless configured as required.
FAIL blocks READY.

## Product behavior

The user sees what failed and what to do next.
