# CP-0026 — Bounce Engine

Purpose:
Detect a candidate ball bounce from the CP-0025 tracked trajectory.

Coordinate convention:
- image Y grows downward
- descending ball: positive vertical velocity
- rising ball: negative vertical velocity

Candidate bounce:
- positive mean vertical velocity before center sample
- negative mean vertical velocity after center sample
- minimum speed thresholds
- refractory period prevents duplicate events
- confidence combines track confidence, raw-observation ratio and reversal strength

Important:
The engine preserves whether the bounce center was based on RAW observation or
TRACKED prediction.

This is deliberately not the final IN/OUT decision engine.
