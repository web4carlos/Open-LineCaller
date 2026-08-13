# CP-0027 — Live Feed Pipeline

MVP goal:
Run the current Open-LineCaller vision chain against a real camera source.

Pipeline:

    Camera / USB webcam / stream
        -> YOLO26 Pickleball v0.2
        -> BallKalmanTracker
        -> CP-0026.2 Curvature Bounce Engine
        -> LIVE overlay

Overlay:
- YOLO bounding box
- tracked x/y
- tracker confidence
- yellow trajectory
- orange prediction during short gaps
- magenta BOUNCE event
- effective pipeline FPS

Controls:
- Q = quit
- R = reset tracker + bounce history

This pack intentionally does NOT include court geometry or IN/OUT.

Those remain:
- CP-0028 Court Geometry
- CP-0029 Decision Engine
