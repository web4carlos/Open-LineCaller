# CP-0014D Technical Design

## Goal

Increase annotation speed while preserving data quality.

## Components

AnnotationKeyframeIndex
- tracks explicit keyframes
- exposes nearest previous/next keyframe

RangeProposalPlanner
- identifies frames eligible for proposal generation
- skips confirmed frames
- limits propagation distance

BatchProposalRunner
- requests proposals over a frame sequence
- returns only eligible frames
- never writes annotations

AccelerationMetrics
- confirmed frames
- proposed frames
- accepted frames
- rejected frames
- skipped confirmed frames
- estimated manual seconds saved

## Safety

No propagation or batch operation may overwrite a confirmed annotation.
