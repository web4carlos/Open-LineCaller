import csv

from linecaller.vision.factory import (
    create_vision_provider,
)
from linecaller.vision.providers.tracknet_csv import (
    TrackNetCsvVisionProvider,
)


def test_factory_tracknet(tmp_path):
    path = tmp_path / "predictions.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Frame", "Visibility", "X", "Y"]
        )

    p = create_vision_provider(
        "tracknet",
        csv_path=path,
    )

    assert isinstance(
        p,
        TrackNetCsvVisionProvider,
    )
