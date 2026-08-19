from __future__ import annotations

import math
import numpy as np
import pytest

from linecaller.flight_paths.continuous_xyz import (
    CalibrationMode,
    ContinuousXYZResolver,
    CourtProjectionCalibration,
    ProjectionMeasurement,
)
from linecaller.flight_paths.models import Point3D, TimedPoint3D


def calibration(mode=CalibrationMode.FULL_COURT):
    # Same point ordering used by LineCaller: NL, NR, FR, FL.
    if mode == CalibrationMode.FULL_COURT:
        pts = np.array([[80, 650], [1180, 650], [820, 120], [420, 120]], dtype=float)
    else:
        # A half-court camera profile still receives one 4-corner visible-region
        # calibration; its Y range is selected by mode, not by a second engine.
        pts = np.array([[70, 650], [1190, 650], [930, 180], [330, 180]], dtype=float)
    return CourtProjectionCalibration(pts, mode)


def measurement_for(cal, p, t, gain=1.7):
    uv = cal.project_xyz(p)
    s = cal.expected_one_bu_scale_px(p.y) * gain
    return ProjectionMeasurement(uv, s, t)


def test_xyz_tern_is_one_point_not_shape_or_path():
    p = Point3D(12.5, 55.0, 9.25)
    assert (p.x, p.y, p.z) == (12.5, 55.0, 9.25)


@pytest.mark.parametrize("mode", [CalibrationMode.FULL_COURT, CalibrationMode.HALF_NEAR, CalibrationMode.HALF_FAR])
def test_full_and_half_court_projection_round_trip(mode):
    cal = calibration(mode)
    y = (cal.y_min_bu + cal.y_max_bu) / 2.0
    p = Point3D(31.0, y, 12.0)
    gain = 1.85
    m = measurement_for(cal, p, 1.0, gain)
    q = cal.unproject_with_scale(m.center_uv_px, m.apparent_scale_px, scale_gain=gain)
    assert q.x == pytest.approx(p.x, abs=0.08)
    assert q.y == pytest.approx(p.y, abs=0.08)
    assert q.z == pytest.approx(p.z, abs=0.08)


def test_json_default_is_full_court_and_half_modes_are_explicit():
    pts = [[0,100],[100,100],[80,0],[20,0]]
    assert CourtProjectionCalibration.from_json_data({"image_points":pts}).mode == CalibrationMode.FULL_COURT
    assert CourtProjectionCalibration.from_json_data({"image_points":pts,"mode":"HALF_NEAR"}).mode == CalibrationMode.HALF_NEAR
    assert CourtProjectionCalibration.from_json_data({"image_points":pts,"calibration_mode":"HALF_FAR"}).mode == CalibrationMode.HALF_FAR


def test_resolver_keeps_same_ball_continuity_and_sub_cell_xyz():
    cal = calibration()
    gain = 1.75
    truth = [
        Point3D(34.20, 72.40, 12.30),
        Point3D(34.85, 70.55, 14.10),
        Point3D(35.55, 68.70, 15.25),
        Point3D(36.30, 66.80, 15.75),
    ]
    times = [10.0, 10.033, 10.066, 10.099]
    r = ContinuousXYZResolver(cal, measurement_weight=0.9)
    first = TimedPoint3D(truth[0], times[0])
    r.prime(first, measurement_for(cal, truth[0], times[0], gain))
    out=[]
    for p,t in zip(truth[1:], times[1:]):
        out.append(r.resolve(measurement_for(cal,p,t,gain)).resolved.point)
    assert out[-1].x == pytest.approx(truth[-1].x, abs=0.35)
    assert out[-1].y == pytest.approx(truth[-1].y, abs=0.35)
    assert out[-1].z == pytest.approx(truth[-1].z, abs=0.35)
    # Continuous result is not forced to integer DCF cells.
    assert not float(out[-1].x).is_integer()
    assert len(r.history) <= 3


def test_scale_gain_is_runtime_nuisance_not_ball_identity():
    cal=calibration()
    p=Point3D(20.0,90.0,8.0)
    t=TimedPoint3D(p,3.0)
    r=ContinuousXYZResolver(cal)
    ev=r.prime(t, measurement_for(cal,p,3.0,gain=2.25))
    assert ev.scale_gain == pytest.approx(2.25, rel=1e-6)
    assert ev.calibration_mode == CalibrationMode.FULL_COURT


def test_time_must_advance():
    cal=calibration()
    p=Point3D(20,90,8)
    r=ContinuousXYZResolver(cal)
    r.prime(TimedPoint3D(p,1.0), measurement_for(cal,p,1.0))
    with pytest.raises(ValueError):
        r.resolve(measurement_for(cal,p,1.0))


def test_projection_measurement_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        ProjectionMeasurement((1.0,2.0), 0.0, 1.0)
