from linecaller.court_geometry import CourtCalibration


def test_calibration_roundtrip(tmp_path):
    c = CourtCalibration(
        image_points=(
            (100, 700),
            (1100, 700),
            (900, 200),
            (300, 200),
        )
    )

    p = tmp_path / "court.json"
    c.save(p)

    loaded = CourtCalibration.load(p)

    assert loaded.image_points == c.image_points
    assert loaded.court_width_ft == 20.0
    assert loaded.court_length_ft == 44.0
