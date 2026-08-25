from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linecaller.dcf.player_net_half_e2e_truth import (
    run_both_player_net_half_e2e_truth_gates,
    run_player_net_half_e2e_truth_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PLAYER NET HALF rendered-frame end-to-end truth gate."
        )
    )
    parser.add_argument(
        "--mount-side",
        choices=("RIGHT", "LEFT", "BOTH"),
        default="BOTH",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    if args.mount_side == "BOTH":
        report = run_both_player_net_half_e2e_truth_gates()
    else:
        result = run_player_net_half_e2e_truth_gate(
            args.mount_side
        )
        report = {
            "feature_version": result.feature_version,
            "passed": result.passed,
            "results": {
                args.mount_side: result.to_dict(),
            },
        }

    print()
    print("Open LineCaller - PLAYER NET HALF E2E TRUTH GATE")
    print("=" * 72)
    for side, result in report["results"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{side:5s} : {status}")
        print(
            "  calls        : "
            f"{result['official_calls']}"
        )
        print(
            "  bootstrap    : "
            f"{result['bootstrap_frames']}"
        )
        print(
            "  ingress      : "
            f"{result['ingress_accepted_frames']}"
        )
        print(
            "  Z0 frames    : "
            f"{result['z0_frames']}"
        )
        print(
            "  UP contacts  : "
            f"{result['up_contact_frames']}"
        )
        print(
            "  deep locks   : "
            f"{result['first_deep_locked_frames']} / "
            f"{result['reacquire_deep_locked_frames']}"
        )
        if result["failures"]:
            for failure in result["failures"]:
                print(f"  FAILURE      : {failure}")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Report        : {path.resolve()}")

    print()
    print(
        "RESULT        : "
        + ("PASS" if report["passed"] else "FAIL")
    )
    print(
        "Physical gate : real phone-on-net footage is still required"
    )
    print()

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
