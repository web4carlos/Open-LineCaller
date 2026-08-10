# CP-0022 Phase 1 — Vision Provider API

Purpose: decouple Open-LineCaller from a specific vision engine.

Contract:

    VisionProvider.detect(...) -> VisionDetection

Current provider: ClassicalVisionProvider.

Future providers can include TrackNet, YOLO, or other sports-vision models.
Phase 1 does not change the live product path.
