from __future__ import annotations

import argparse
import json

from linecaller.dataset.project import DatasetProject
from linecaller.dataset.splitting import split_clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = DatasetProject(args.root)
    manifest = project.load_manifest()

    split = split_clips(
        manifest.clips,
        seed=args.seed,
    )

    project.splits_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": args.seed,
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }

    path = project.splits_dir / "split.json"
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(path)


if __name__ == "__main__":
    main()
