# CP-0022 Phase 2 — TrackNet Evaluation Provider

## Decision

TrackNet runs in a separate environment from Open-LineCaller.

Reason:
The selected TrackNet Series implementation maintains a Python 3.11 workspace,
while the existing Open-LineCaller development environment can remain unchanged.

## Data bridge

TrackNet inference exports:

    Frame,Visibility,X,Y

in original-video coordinate space.

Open-LineCaller consumes that file through:

    TrackNetCsvVisionProvider

This is deliberately an offline evaluation bridge.

## Why not native LIVE yet?

We have not demonstrated that:
- tennis/badminton-trained weights track pickleball well;
- latency is suitable for LIVE;
- the chosen checkpoint is the correct model.

Native runtime coupling would be premature.

## Phase 2 success criteria

1. TrackNet Series environment installs independently.
2. A trained checkpoint runs against MATCH001.
3. TrackNet produces predictions CSV and overlay MP4.
4. Open-LineCaller reads the CSV through VisionProvider.
5. Visual overlay is inspected against the actual pickleball.

If TrackNet follows the ball well, Phase 3 can benchmark it formally and design
a streaming provider.
