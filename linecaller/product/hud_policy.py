from __future__ import annotations

from .hud_models import (
    HUDMetric,
    LastCallPresentation,
    LatencyQuality,
    StatusLevel,
)


def tracking_metric(status: str) -> HUDMetric:
    normalized = str(status or "SEARCHING").upper()

    if normalized == "LOCKED":
        return HUDMetric("Tracking", "LOCKED", StatusLevel.OK, "Stable track")

    if normalized in ("TRACK FOUND", "REACQUIRING"):
        return HUDMetric("Tracking", normalized, StatusLevel.WARN, "Stabilizing")

    if normalized == "LOST":
        return HUDMetric("Tracking", "LOST", StatusLevel.ERROR, "No reliable track")

    return HUDMetric("Tracking", "SEARCHING", StatusLevel.WARN, "Looking for ball")


def fps_metric(fps: float) -> HUDMetric:
    fps = float(fps)

    if fps >= 50:
        return HUDMetric("FPS", f"{fps:.1f}", StatusLevel.OK, "Excellent")

    if fps >= 30:
        return HUDMetric("FPS", f"{fps:.1f}", StatusLevel.WARN, "Acceptable")

    if fps > 0:
        return HUDMetric("FPS", f"{fps:.1f}", StatusLevel.ERROR, "Too low")

    return HUDMetric("FPS", "0.0", StatusLevel.UNKNOWN, "No data")


def latency_quality(latency_ms: float) -> LatencyQuality:
    latency_ms = float(latency_ms)

    if latency_ms <= 0:
        return LatencyQuality.UNKNOWN
    if latency_ms <= 60:
        return LatencyQuality.EXCELLENT
    if latency_ms <= 120:
        return LatencyQuality.ACCEPTABLE
    return LatencyQuality.HIGH


def latency_metric(latency_ms: float) -> HUDMetric:
    quality = latency_quality(latency_ms)

    mapping = {
        LatencyQuality.EXCELLENT: (StatusLevel.OK, "Excellent"),
        LatencyQuality.ACCEPTABLE: (StatusLevel.WARN, "Acceptable"),
        LatencyQuality.HIGH: (StatusLevel.ERROR, "Too High"),
        LatencyQuality.UNKNOWN: (StatusLevel.UNKNOWN, "No data"),
    }

    level, detail = mapping[quality]

    return HUDMetric(
        "Latency",
        f"{float(latency_ms):.1f} ms",
        level,
        detail,
    )


def confidence_metric(confidence: float) -> HUDMetric:
    confidence = max(0.0, min(1.0, float(confidence)))

    if confidence >= .98:
        level = StatusLevel.OK
        detail = "High confidence"
    elif confidence >= .90:
        level = StatusLevel.WARN
        detail = "Review quality"
    elif confidence > 0:
        level = StatusLevel.ERROR
        detail = "Low confidence"
    else:
        level = StatusLevel.UNKNOWN
        detail = "No data"

    return HUDMetric(
        "Confidence",
        f"{confidence*100:.1f}%",
        level,
        detail,
    )


def replay_metric(ready: bool) -> HUDMetric:
    return HUDMetric(
        "Replay",
        "READY" if ready else "IDLE",
        StatusLevel.OK if ready else StatusLevel.UNKNOWN,
        "Instant replay available" if ready else "Waiting",
    )


def last_call_presentation(call: str) -> LastCallPresentation:
    call = str(call or "-").upper()

    if call == "IN":
        return LastCallPresentation(
            "IN",
            "Ball landed in",
            StatusLevel.OK,
        )

    if call == "OUT":
        return LastCallPresentation(
            "OUT",
            "Ball landed out",
            StatusLevel.ERROR,
        )

    if call == "REVIEW":
        return LastCallPresentation(
            "REVIEW",
            "Instant replay",
            StatusLevel.WARN,
        )

    return LastCallPresentation()
