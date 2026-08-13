# CP-0031 — Full Court Semantics

Purpose:
Upgrade Open-LineCaller from a four-boundary rectangle to a semantic
pickleball court model.

Court coordinates:
- x = 0..20 ft
- y = 0..44 ft
- FAR baseline = y 0
- FAR NVZ line = y 15
- NET = y 22
- NEAR NVZ line = y 29
- NEAR baseline = y 44
- center x = 10

Semantic zones:
FAR_LEFT_SERVICE, FAR_RIGHT_SERVICE, FAR_KITCHEN,
NEAR_LEFT_SERVICE, NEAR_RIGHT_SERVICE, NEAR_KITCHEN,
NET, OUTSIDE.

Semantic lines:
LEFT_SIDELINE, RIGHT_SIDELINE, FAR_BASELINE, NEAR_BASELINE,
FAR_NVZ_LINE, NEAR_NVZ_LINE, FAR_CENTERLINE, NEAR_CENTERLINE, NET.

Centerlines exist only between each baseline and its NVZ line.
CP-0032 will apply pickleball rules to these semantic zones.
