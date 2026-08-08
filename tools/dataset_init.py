from __future__ import annotations

import argparse

from linecaller.dataset.project import DatasetProject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name")
    parser.add_argument("--version", default="1.0")
    args = parser.parse_args()

    project = DatasetProject(args.root)
    project.initialize(name=args.name, version=args.version)

    print(f"Dataset initialized: {project.root}")
    print(f"Manifest: {project.manifest_path}")


if __name__ == "__main__":
    main()
