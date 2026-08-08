from linecaller.dataset.models import (
    BallBox,
    BounceAnnotation,
    ClipAnnotation,
    DatasetManifest,
    FrameAnnotation,
)
from linecaller.dataset.project import DatasetProject


def test_annotation_roundtrip(tmp_path):
    p = DatasetProject(tmp_path)
    p.initialize()

    ann = ClipAnnotation(
        clip_id="c1",
        source_video="c1.mp4",
        width=640,
        height=480,
        fps=60,
        frame_count=100,
        frames=[
            FrameAnnotation(
                frame_number=5,
                ball=BallBox(10,20,8,8),
                visible=True,
                occluded=False,
            )
        ],
        bounces=[
            BounceAnnotation(
                frame_number=20,
                x=50,
                y=60,
                decision="OUT",
            )
        ],
    )

    p.save_clip_annotation(ann)
    loaded = p.load_clip_annotation("c1")

    assert loaded.clip_id == "c1"
    assert loaded.frames[0].ball.width == 8
    assert loaded.bounces[0].decision == "OUT"
