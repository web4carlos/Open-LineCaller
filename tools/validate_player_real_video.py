from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linecaller.dcf.player_real_video_validation import (
    ExpectedOutCall,
    validate_player_real_video,
)


def _expected_out(value: str) -> ExpectedOutCall:
    try:
        frame_text, region = value.split(":", 1)
        frame = int(frame_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "expected OUT must be FRAME:REGION, "
            "for example 147:OUT_LEFT"
        ) from exc
    if frame < 0:
        raise argparse.ArgumentTypeError(
            "expected OUT frame must be >= 0"
        )
    try:
        return ExpectedOutCall(frame, region)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a real Player phone-on-net video through the "
            "official outside-only LineCaller runtime."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--ball-template", required=True)
    parser.add_argument("--background", default=None)
    parser.add_argument(
        "--mount-side",
        choices=("RIGHT", "LEFT"),
        required=True,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--expect-out",
        action="append",
        type=_expected_out,
        default=[],
        metavar="FRAME:REGION",
        help=(
            "Optional truth expectation; repeat for multiple OUTs. "
            "Example: --expect-out 147:OUT_LEFT"
        ),
    )
    parser.add_argument(
        "--expect-no-out",
        action="store_true",
        help="Truth mode: selected frame range must emit zero OUT calls.",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=int,
        default=2,
        help="Allowed contact-frame truth tolerance (default: 2).",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=120,
    )
    args = parser.parse_args()

    if args.expect_out and args.expect_no_out:
        parser.error(
            "--expect-out and --expect-no-out are mutually exclusive"
        )
    if args.contact_tolerance < 0:
        parser.error("--contact-tolerance must be >= 0")
    if args.max_keyframes < 0:
        parser.error("--max-keyframes must be >= 0")

    video = Path(args.video).expanduser().resolve()
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = (
            REPO_ROOT
            / ".linecaller_runtime"
            / "validation"
            / video.stem
        ).resolve()

    try:
        report = validate_player_real_video(
            video_path=video,
            calibration_path=args.calibration,
            ball_template_path=args.ball_template,
            background_path=args.background,
            mount_side=args.mount_side,
            output_dir=output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            expected_calls=tuple(args.expect_out),
            expect_no_out=bool(args.expect_no_out),
            contact_tolerance_frames=args.contact_tolerance,
            max_keyframes=args.max_keyframes,
        )
    except Exception as exc:
        print()
        print("PLAYER REAL-VIDEO VALIDATION: ERROR")
        print("=" * 72)
        print(str(exc))
        print()
        return 2

    print()
    print("Open LineCaller - PLAYER REAL-VIDEO VALIDATION")
    print("=" * 72)
    print(f"Status         : {report.status}")
    print(f"Mount          : {report.mount_side}")
    print(f"Frames         : {report.processed_frames}")
    print(f"FPS            : {report.fps:.3f}")
    print(f"Image size     : {report.image_size}")
    print(f"Pose           : {report.pose_counts}")
    print(
        "Ball frames    : "
        f"{report.frames_with_ball_components}"
    )
    print(f"Bootstrap      : {report.bootstrap_frames}")
    print(f"Ingress ACCEPT : {report.ingress_accepted_frames}")
    print(f"Z0 frames      : {report.z0_frames}")
    print(f"UP frames      : {report.up_frames}")
    print(
        "Official calls : "
        + (
            str(
                [
                    (
                        call.contact_frame,
                        call.region,
                        call.confirm_frame,
                    )
                    for call in report.official_calls
                ]
            )
            if report.official_calls
            else "[]"
        )
    )
    print(f"Warnings       : {list(report.warnings)}")
    print(f"Report         : {report.report_json}")
    print(f"Timeline JSONL : {report.timeline_jsonl}")
    print(f"Timeline CSV   : {report.timeline_csv}")
    print(f"Keyframes      : {report.keyframe_dir}")

    if report.truth_evaluated:
        print(
            "Truth          : "
            + ("PASS" if report.truth_passed else "FAIL")
        )
        if report.truth_failures:
            for failure in report.truth_failures:
                print(f"  FAILURE      : {failure}")
    else:
        print(
            "Truth          : NOT PROVIDED "
            "(diagnostic run; no accuracy claim)"
        )

    print()
    if report.truth_evaluated and not report.truth_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
