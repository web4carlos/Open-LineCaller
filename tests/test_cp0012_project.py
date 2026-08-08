from linecaller.dataset.project import DatasetProject


def test_initialize_project(tmp_path):
    root = tmp_path / "dataset"
    p = DatasetProject(root)

    p.initialize(name="test")

    assert p.manifest_path.exists()
    assert p.annotations_dir.exists()
    assert p.clips_dir.exists()

    manifest = p.load_manifest()
    assert manifest.name == "test"
