from __future__ import annotations

import argparse
import json

from linecaller.dataset.project import DatasetProject
from linecaller.dataset.reporting import dataset_quality_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    project = DatasetProject(args.root)
    report = dataset_quality_report(project)

    print(json.dumps(report, indent=2))

    if report["errors"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
