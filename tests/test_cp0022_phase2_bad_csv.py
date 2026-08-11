import pytest

from linecaller.vision.providers.tracknet_csv import (
    TrackNetCsvVisionProvider,
)


def test_bad_csv_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "frame,x,y\n0,1,2\n",
        encoding="utf-8",
    )

    p = TrackNetCsvVisionProvider(
        csv_path=path
    )

    with pytest.raises(ValueError):
        p.initialize()
