# CP-0014A Technical Design — Proposal Engine API

## Goal

Decouple annotation UI and video pipeline from concrete detector implementations.

## Canonical proposal

BallProposal contains:

- frame_number
- x / y
- width / height
- confidence
- source
- status
- optional metadata

Coordinates are source-image pixels.

## Interfaces

ProposalEngine
    propose(frame_number, frame) -> ProposalResult

ProposalResult
    proposals
    latency_ms
    engine_name

## Fusion

ProposalFusion accepts results from multiple engines.

Initial policy:
- discard invalid boxes
- normalize confidence to [0,1]
- cluster by center distance / IoU
- keep strongest proposal per cluster
- optionally boost agreement between independent engines

## Safety

The proposal layer never writes annotations by itself.

The human annotation workflow decides whether to:
- accept
- adjust
- reject

## Future engines

- YOLOProposalEngine
- ONNXProposalEngine
- TensorRTProposalEngine
- MotionProposalEngine
- EnsembleProposalEngine
