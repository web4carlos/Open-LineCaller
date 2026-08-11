import csv
import numpy as np

from linecaller.vision.providers.tracknet_csv import (
    TrackNetCsvVisionProvider,
)


def make_csv(path):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Frame", "Visibility", "X", "Y"]
        )
        writer.writerow([0, 1, 100, 200])
        writer.writerow([1, 0, -1, -1])
        writer.writerow([2, 1, 120, 220])


def test_tracknet_csv_provider(tmp_path):
    path = tmp_path / "predictions.csv"
    make_csv(path)

    p = TrackNetCsvVisionProvider(
        csv_path=path
    )

    frame = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    a = p.detect(
        frame=frame,
        frame_number=0,
        timestamp=0.0,
    )
    b = p.detect(
        frame=frame,
        frame_number=1,
        timestamp=.1,
    )

    assert a.x == 100
    assert a.y == 200
    assert a.status == "TRACKING"

    assert b.x is None
    assert b.y is None
    assert b.status == "LOST"
