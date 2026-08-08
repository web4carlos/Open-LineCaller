import json

from linecaller.benchmark.manifest import load_manifest


def test_load_manifest(tmp_path):
    manifest = {
        "clips": [{
            "id":"a",
            "video":"videos/a.mp4",
            "calibration":"calibrations/a.json",
            "expected":"expected/a.json"
        }]
    }

    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    clips = load_manifest(tmp_path)

    assert len(clips) == 1
    assert clips[0].clip_id == "a"
    assert clips[0].video.name == "a.mp4"
