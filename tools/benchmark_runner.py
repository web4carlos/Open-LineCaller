from __future__ import annotations

import argparse

from linecaller.benchmark.runner import BenchmarkRunner
from linecaller.benchmark.reporting import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--frame-tolerance", type=int, default=2)
    args = parser.parse_args()

    runner = BenchmarkRunner(frame_tolerance=args.frame_tolerance)
    report = runner.run(args.root)

    json_path, txt_path = write_report(report, args.root)

    print("")
    print(txt_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")


if __name__ == "__main__":
    main()
