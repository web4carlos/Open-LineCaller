# CP-0015 Technical Design — Live Officiating Core

## Goal

Create deterministic low-latency infrastructure for live pickleball officiating.

## State flow

LIVE
  -> event arrives
  -> policy evaluates event
      -> IN
      -> OUT
      -> REVIEW
  -> REVIEW requests replay
  -> cooldown suppresses duplicate calls
  -> evidence record persists decision metrics

## Replay

A bounded ring buffer stores recent frames.

When REVIEW is triggered, the engine snapshots:
- configurable frames before the event
- the event frame
- configurable post-event frames when available

CP-0016 will expose replay visually and add voice calls.

## Latency

Every processed frame can record:
- capture timestamp
- processing start
- processing end

Event evidence records:
- frame number
- decision
- confidence
- decision latency
- processing FPS
- replay requested

## Accuracy Gate

AccuracyGate is a release/readiness metric, not a decision rule.

PASS:
    automatic-call accuracy >= 0.98
    and minimum validation sample count is satisfied

REVIEW events are reported separately and are not silently counted as correct
automatic calls.

## Safety

- duplicate calls are suppressed by cooldown
- REVIEW never becomes IN or OUT automatically
- low-confidence evidence can be routed to replay
- live metrics do not alter geometric decision logic
