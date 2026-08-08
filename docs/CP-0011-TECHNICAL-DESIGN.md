# CP-0011 Technical Design — Benchmark Suite

## Goal

Provide stable, reproducible metrics across a fixed benchmark corpus.

## Inputs

Manifest
  -> video path
  -> calibration profile
  -> expected bounce events

## Execution

For each clip:

1. run IntegratedVideoPipeline
2. parse generated JSONL decisions
3. compare detected bounce frames to expected bounce frames
4. match events within configurable frame tolerance
5. compute:
   - TP
   - FP
   - FN
   - precision
   - recall
   - timing error
   - decision agreement
6. aggregate over all clips

## Matching

Greedy nearest-frame matching is used initially with a configurable tolerance.

A detection can match at most one expected event.

## Reproducibility

Reports include:
- timestamp
- benchmark root
- clip IDs
- tolerance
- aggregate metrics

## Future extensions

- stratified metrics by indoor/outdoor
- motion-blur category
- camera angle
- distance-to-line bucket
- confidence calibration curves
