# CP-0028 — Court Geometry

Goal:
Convert bounce pixel coordinates into real pickleball-court coordinates.

Court dimensions:
- width: 20 ft
- length: 44 ft

Calibration:
4 clicks only.

Click order:
1. near-left
2. near-right
3. far-right
4. far-left

Outputs per bounce:
- court_x_ft
- court_y_ft
- inside_court
- nearest_line
- signed_distance_ft
- absolute_distance_ft
- geometry_state:
  - INSIDE
  - OUTSIDE
  - NEAR_LINE

Important:
This is geometry only.
CP-0029 will make the official IN / OUT / REVIEW decision.
