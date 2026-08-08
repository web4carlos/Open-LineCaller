# CP-0017 Technical Design — Live Pipeline Integration

## Purpose

CP-0016 created the user-facing live shell.
CP-0017 connects that shell to the decision path.

## New boundary

The UI does not call detector/tracker/bounce modules directly.

It calls:

    LivePerceptionAdapter.process(frame_number, frame, timestamp)

which returns:

    LivePipelineResult

The result can contain:
- tracking status
- detection confidence
- ball center
- bounce detected
- line decision
- decision confidence
- metadata

## Why an adapter?

The existing Open-LineCaller modules evolved independently across capability
packs. The adapter provides one stable real-time contract while allowing the
underlying implementation to improve later.

## Event conversion

Only when a result contains a decision does the adapter emit a LiveEvent.

Allowed decisions:
- IN
- OUT
- REVIEW

REVIEW is not converted into a final call.

## Latency

Processing time is measured around each adapter invocation.
The LiveMatchController receives:
- current FPS
- processing latency
- tracking state
- confidence

## Safety

No result => no call.
Lost tracking => no call.
Duplicate events => suppressed by CP-0015 cooldown.
