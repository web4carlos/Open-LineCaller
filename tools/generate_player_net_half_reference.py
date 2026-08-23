from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# When this tool is executed as `python tools/...py`, Python places the
# tools directory (not the repository root) on sys.path. Add the repo root
# explicitly so the sibling `linecaller` package is importable on Windows
# and on normal direct-script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linecaller.dcf.player_net_half_synthetic import (
    PlayerNetHalfSyntheticReference,
    run_net_ingress_selector_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic PLAYER NET HALF synthetic reference."
    )
    parser.add_argument(
        "--output",
        default=".linecaller_runtime/synthetic/player_net_half",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args()

    gate = run_net_ingress_selector_gate()
    if not gate["passed"]:
        print(json.dumps(gate, indent=2))
        raise SystemExit("Synthetic net-ingress selector gate FAILED.")

    bundle = PlayerNetHalfSyntheticReference().build_bundle(
        Path(args.output),
        write_video=not args.no_video,
        write_frames=not args.no_frames,
    )

    print()
    print("Open LineCaller - PLAYER NET HALF synthetic reference")
    print("=" * 72)
    print("Selector gate : PASS")
    print(f"Output        : {Path(bundle.output_dir).resolve()}")
    print(f"Frames        : {bundle.frame_count}")
    print(f"FPS           : {bundle.fps}")
    print(f"Calibration   : {Path(bundle.calibration).resolve()}")
    print(f"Background    : {Path(bundle.background).resolve()}")
    print(f"Ball template : {Path(bundle.ball_template).resolve()}")
    print(f"Truth         : {Path(bundle.truth).resolve()}")
    print(
        "Video         : "
        + (
            str(Path(bundle.video).resolve())
            if bundle.video is not None
            else "codec unavailable; PNG/core assets remain valid"
        )
    )
    print("Truth contacts: frame 24 -> OUT_LEFT, frame 52 -> IN")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
