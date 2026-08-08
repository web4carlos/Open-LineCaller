from linecaller.dataset.models import (
    BallBox,
    BounceAnnotation,
    ClipAnnotation,
    FrameAnnotation,
)
from linecaller.dataset.validation import validate_clip


def test_valid_annotation_has_no_errors():
    clip = ClipAnnotation(
        clip_id="x",
        source_video="x.mp4",
        width=1920,
        height=1080,
        fps=60,
        frame_count=100,
        frames=[
            FrameAnnotation(
                frame_number=10,
                ball=BallBox(100,100,20,20),
            )
        ],
        bounces=[
            BounceAnnotation(
                frame_number=20,
                x=200,
                y=300,
                decision="IN",
            )
        ],
    )

    issues = validate_clip(clip)
    assert not [i for i in issues if i.severity == "ERROR"]


def test_invalid_decision_is_error():
    clip = ClipAnnotation(
        clip_id="x",
        source_video="x.mp4",
        width=100,
        height=100,
        fps=30,
        frame_count=10,
        bounces=[
            BounceAnnotation(
                frame_number=5,
                x=1,
                y=1,
                decision="MAYBE",
            )
        ],
    )

    issues = validate_clip(clip)
    assert any("Invalid decision" in i.message for i in issues)
