# CP-0018 Sprint 5 Technical Design — Replay Experience

## Goal

Turn REVIEW into a smooth visual sequence:

LIVE -> FREEZE -> REPLAY -> RESUME -> LIVE

## Components

ReplayState
- LIVE
- FREEZE
- PLAYING
- RESUME

ReplayPlaybackController
- loads buffered frame packets
- keeps decision frame index
- controls speed
- advances playback
- returns to LIVE when complete

ReplayViewModel
- display frame number
- speed label
- decision marker
- zoom center
- active flag

ReplayMetrics
- replay count
- total frames shown
- average frames per replay

## Safety

Replay never changes a decision.
Replay is presentation only.

## Integration

LiveMatchWindow switches its preview source from live frames to replay frames
only while replay is active.
