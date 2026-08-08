import cv2
import numpy as np

from linecaller.pipeline.video_source import VideoSource


def test_video_source_roundtrip(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    for i in range(5):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    source = VideoSource(path)
    source.open()

    frames = list(source.frames())

    assert len(frames) == 5
    assert source.width == 64
    assert source.height == 48

    source.close()
