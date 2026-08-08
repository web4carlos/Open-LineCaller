from __future__ import annotations

from dataclasses import dataclass

from .models import ClipAnnotation


@dataclass(frozen=True)
class ValidationIssue:
    clip_id: str
    severity: str
    message: str


def validate_clip(annotation: ClipAnnotation) -> tuple[ValidationIssue, ...]:
    issues = []

    if annotation.width <= 0 or annotation.height <= 0:
        issues.append(
            ValidationIssue(annotation.clip_id, "ERROR", "Invalid video dimensions")
        )

    if annotation.fps <= 0:
        issues.append(
            ValidationIssue(annotation.clip_id, "ERROR", "FPS must be > 0")
        )

    if annotation.frame_count <= 0:
        issues.append(
            ValidationIssue(annotation.clip_id, "ERROR", "Frame count must be > 0")
        )

    seen_frames = set()

    for frame in annotation.frames:
        if frame.frame_number in seen_frames:
            issues.append(
                ValidationIssue(
                    annotation.clip_id,
                    "ERROR",
                    f"Duplicate frame annotation: {frame.frame_number}",
                )
            )
        seen_frames.add(frame.frame_number)

        if not (0 <= frame.frame_number < annotation.frame_count):
            issues.append(
                ValidationIssue(
                    annotation.clip_id,
                    "ERROR",
                    f"Frame out of range: {frame.frame_number}",
                )
            )

        if frame.ball is not None:
            for err in frame.ball.validate():
                issues.append(
                    ValidationIssue(annotation.clip_id, "ERROR", err)
                )

            b = frame.ball
            if b.x < 0 or b.y < 0:
                issues.append(
                    ValidationIssue(
                        annotation.clip_id,
                        "WARNING",
                        f"Ball box starts outside image on frame {frame.frame_number}",
                    )
                )

            if b.x + b.width > annotation.width or b.y + b.height > annotation.height:
                issues.append(
                    ValidationIssue(
                        annotation.clip_id,
                        "WARNING",
                        f"Ball box exceeds image bounds on frame {frame.frame_number}",
                    )
                )

    for bounce in annotation.bounces:
        if not (0 <= bounce.frame_number < annotation.frame_count):
            issues.append(
                ValidationIssue(
                    annotation.clip_id,
                    "ERROR",
                    f"Bounce frame out of range: {bounce.frame_number}",
                )
            )

        if bounce.decision not in (None, "IN", "OUT", "REVIEW"):
            issues.append(
                ValidationIssue(
                    annotation.clip_id,
                    "ERROR",
                    f"Invalid decision label: {bounce.decision}",
                )
            )

    return tuple(issues)
