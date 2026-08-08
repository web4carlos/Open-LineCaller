# CP-0009 Technical Design

## Goal

Reduce false court proposals by comparing multiple hypotheses.

## Hypothesis score components

- quadrilateral area ratio
- line support
- opposite-side length consistency
- convexity
- image-bound plausibility
- orientation-family separation

## Ambiguity rule

Even a high-scoring best hypothesis should not be AUTO_ACCEPT when the
runner-up score is too close.

The confidence margin between #1 and #2 is therefore part of the final
disposition.

## Safety

AUTO_ACCEPT means "safe to prefill calibration points", not "safe to officiate".
Final officiating still requires the CP-0002 calibration quality gate.
