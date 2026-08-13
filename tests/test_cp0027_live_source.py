from linecaller.live import parse_video_source


def test_camera_zero_is_integer():
    assert parse_video_source("0") == 0


def test_camera_one_is_integer():
    assert parse_video_source("1") == 1


def test_video_path_stays_string():
    value = r"C:\validation\match.mp4"
    assert parse_video_source(value) == value
