import numpy as np

from linecaller.product.replay_render import replay_zoom


def test_replay_zoom_preserves_output_size():
    frame = np.zeros((100,200,3), dtype=np.uint8)

    out = replay_zoom(
        frame,
        x=100,
        y=50,
        scale=2.0,
    )

    assert out.shape == frame.shape


def test_replay_zoom_without_center_returns_frame():
    frame = np.zeros((20,20,3), dtype=np.uint8)

    out = replay_zoom(frame)

    assert out is frame
