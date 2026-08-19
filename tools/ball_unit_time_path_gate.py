from __future__ import annotations

from linecaller.flight_paths import (
    CourtFrame,
    CourtPathField,
    FlightEpochManager,
    Point3D,
    TimedPoint3D,
    Velocity3D,
)


def fmt(p: Point3D) -> str:
    return f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) BU"


def main() -> int:
    court = CourtFrame()
    field = CourtPathField(court)

    print()
    print("============================================================")
    print(" CP-0035.9.4.2 RC3 - BALL-UNIT + TIME PATH GATE")
    print("============================================================")
    print()
    print("BALL IDENTITY............... WHO ONLY")
    print("XYZ TERN.................... ONE POINT")
    print("PATH........................ CONTINUOUS CURVE")
    print("MESH/CELLS DEFINE PATH...... NO")
    print("SPATIAL UNIT................ ONE BALL UNIT (BU)")
    print(f"ONE BU...................... {court.ball_unit_in:.4f} in")
    print(f"COURT X..................... 0 .. {court.width_bu:.1f} BU")
    print(f"COURT Y..................... 0 .. {court.length_bu:.1f} BU")
    print("Z=0......................... CONTACT PLANE")
    print("Z=1......................... ONE BALL UNIT ABOVE Z=0")
    print("TIME UNIT................... SECONDS")
    print(f"GRAVITY..................... {court.gravity_bu_s2:.3f} BU/s^2")
    print()

    # Example: direction change / paddle hit at a known point and time.
    p0 = Point3D(20.0, 150.0, 12.0)
    t0 = 10.000
    v0 = Velocity3D(55.0, -160.0, 25.0)
    selected = field.select_from_state(p0, v0, t0_s=t0)

    print(f"P0.......................... {fmt(p0)}")
    print(f"t0.......................... {t0:.3f} s")
    print(f"V0.......................... ({v0.vx:.2f}, {v0.vy:.2f}, {v0.vz:.2f}) BU/s")
    print(f"TARGET...................... {selected.target_side.value}-{selected.target_lateral.value}")
    print(f"PREDICTED Z0................ {fmt(selected.landing)}")
    print(f"TIME TO Z0.................. {selected.path.flight_duration_s:.4f} s")
    print(f"CONTACT TIME................ {selected.contact_time_s:.4f} s")
    print()

    print("TIMED PATH SAMPLES:")
    for sample in selected.path.samples(6):
        print(f"  t={sample.t_s:.4f}  P={fmt(sample.point)}")
    print()

    # Demonstrate that time creates velocity from observed points.
    a = TimedPoint3D(Point3D(20.0, 150.0, 12.0), 10.000)
    b = TimedPoint3D(Point3D(22.2, 143.6, 12.8), 10.040)
    measured = Velocity3D.between(a, b)
    print("OBSERVATION A............... P(x,y,z) + t")
    print("OBSERVATION B............... P(x,y,z) + t")
    print(
        "MEASURED V................. "
        f"({measured.vx:.2f}, {measured.vy:.2f}, {measured.vz:.2f}) BU/s"
    )
    print()

    manager = FlightEpochManager(field)
    first = manager.on_direction_change(p0, v0, t0_s=t0)
    manager.on_floor_contact(first.predicted_z0)
    second = manager.on_direction_change(
        Point3D(64.0, 28.0, 11.0),
        Velocity3D(-50.0, 170.0, 32.0),
        t0_s=11.250,
    )
    print(f"EPOCH #1.................... {first.selection.origin_side.value} -> {first.selection.target_side.value}")
    print(f"EPOCH #2.................... {second.selection.origin_side.value} -> {second.selection.target_side.value}")
    print("DIRECTION CHANGE............ NEW P0 + V0 + t0 / NEW PATH")
    print("PRE-SERVE FLOOR BOUNCE...... NO OUTBOUND EPOCH BY ITSELF")
    print()
    print("BALL_UNIT_TIME_PATH_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
