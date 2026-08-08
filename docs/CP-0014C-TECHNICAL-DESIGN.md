# CP-0014C Technical Design — YOLO Proposal Engine

## Goal

Add a real neural object detector without coupling the rest of Open-LineCaller
to Ultralytics.

## Boundary

Dataset Studio depends on:

    ProposalEngine

not:

    ultralytics.YOLO

YOLOProposalEngine owns the integration boundary.

## Inference

The engine calls the injected backend/model with:

- source frame
- confidence threshold
- verbose=False

It expects detection Results objects whose `boxes` expose:

- xyxy
- conf
- cls

This matches the current Ultralytics detection-results API.

## Class filtering

Filter can be:
- class name (`sports ball`)
- no filter (`None`)

Names are resolved from result.names or model.names.

## Resilience

Model loading is lazy.

If Ultralytics is unavailable or model loading fails:
- the engine can return an empty ProposalResult when `fail_open=True`
- metadata contains an error message
- Dataset Studio remains usable manually / temporally

## Testing

Unit tests inject a fake YOLO backend and fake Results objects.

No network, GPU, model download, or Ultralytics inference is required during
the test suite.

## Future

Our custom model will use the same API:

    YOLOProposalEngine(weights="models/open-linecaller-ball-v1.pt",
                       class_name="pickleball")
