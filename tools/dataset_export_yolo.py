from __future__ import annotations

import argparse

from linecaller.dataset.export_yolo import export_clip_yolo
from linecaller.dataset.project import DatasetProject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = DatasetProject(args.root)
    manifest = project.load_manifest()

    total = 0

    for clip_id in manifest.clips:
        annotation = project.load_clip_annotation(clip_id)
        total += export_clip_yolo(annotation, args.output)

    print(f"YOLO label files written: {total}")


if __name__ == "__main__":
    main()
