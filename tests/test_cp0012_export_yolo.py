from linecaller.dataset.export_yolo import export_clip_yolo
from linecaller.dataset.models import (
    BallBox,
    ClipAnnotation,
    FrameAnnotation,
)


def test_export_yolo(tmp_path):
    clip = ClipAnnotation(
        clip_id="clip1",
        source_video="clip1.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=10,
        frames=[
            FrameAnnotation(
                frame_number=1,
                ball=BallBox(40,40,20,20),
            )
        ],
    )

    count = export_clip_yolo(clip, tmp_path)

    assert count == 1

    text = (
        tmp_path
        / "labels"
        / "clip1_000001.txt"
    ).read_text().strip()

    assert text.startswith("0 ")
    parts = text.split()
    assert float(parts[1]) == 0.5
    assert float(parts[2]) == 0.5
