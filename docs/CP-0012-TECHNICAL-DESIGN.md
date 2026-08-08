# CP-0012 Technical Design — Vision Dataset Foundation

## Goal

Create a stable, model-agnostic data layer for future pickleball vision training.

## Dataset hierarchy

datasets/<dataset_name>/
  manifest.json
  clips/
  annotations/
  splits/
  exports/
  reports/

## Annotation model

Per clip:

- clip_id
- source_video
- width
- height
- fps
- frame_count
- frame annotations:
  - frame number
  - ball bounding box
  - visibility
  - occlusion
  - confidence / quality flag
- bounce annotations:
  - frame number
  - x/y image point
  - decision label
  - optional court coordinates

## Design rules

1. Original source video is never modified.
2. Annotation data is separate from media.
3. Splits are deterministic when a seed is provided.
4. Exporters consume the same canonical annotation model.
5. Quality checks run before training export.

## Future extensions

- multiple balls / objects
- player occlusion labels
- line masks
- court segmentation
- frame extraction cache
- active learning
- model-assisted annotation
