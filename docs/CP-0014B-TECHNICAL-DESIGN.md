# CP-0014B Technical Design

## Goal

Reduce annotation effort while keeping humans in control.

## Main components

AssistedAnnotationController
- requests proposals
- tracks proposal state
- accepts / rejects / adjusts
- records productivity metrics

TemporalProposalTracker
- predicts next-frame proposal from recent accepted boxes
- constant-velocity center model
- box size smoothing
- never writes annotations directly

Dataset Studio integration
- confirmed annotation = green
- proposal = yellow / red / cyan depending on status
- confirmed annotation always wins
- proposals never overwrite existing confirmed annotations

## Safety rule

If the current frame already has a confirmed ball box, no automatic proposal
may replace it.

## Metrics

- proposals_requested
- proposals_shown
- accepted
- adjusted
- rejected
- auto_advanced
- acceptance_rate
