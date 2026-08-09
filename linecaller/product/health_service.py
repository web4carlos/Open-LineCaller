from __future__ import annotations

import shutil
from pathlib import Path

from .health_models import (
    HealthCheckItem,
    HealthReport,
    HealthStatus,
)


class HealthCheckService:
    def __init__(
        self,
        *,
        min_fps: float = 30.0,
        max_latency_ms: float = 100.0,
        min_free_gb: float = 1.0,
    ):
        self.min_fps = float(min_fps)
        self.max_latency_ms = float(max_latency_ms)
        self.min_free_gb = float(min_free_gb)

    def evaluate(
        self,
        *,
        camera_ok: bool,
        calibration_ok: bool,
        replay_ok: bool,
        audio_ok: bool,
        tracking_status: str,
        fps: float,
        latency_ms: float,
        storage_path: str | Path = ".",
    ) -> HealthReport:
        items = [
            self._camera(camera_ok),
            self._calibration(calibration_ok),
            self._replay(replay_ok),
            self._audio(audio_ok),
            self._tracking(tracking_status),
            self._fps(fps),
            self._latency(latency_ms),
            self._storage(storage_path),
        ]

        return HealthReport(tuple(items))

    def _camera(self, ok):
        return HealthCheckItem(
            "Camera",
            HealthStatus.PASS if ok else HealthStatus.FAIL,
            "Connected" if ok else "Camera not available",
            "" if ok else "Reconnect camera or choose another camera.",
        )

    def _calibration(self, ok):
        return HealthCheckItem(
            "Calibration",
            HealthStatus.PASS if ok else HealthStatus.FAIL,
            "Valid" if ok else "Calibration required",
            "" if ok else "Run Auto, Assisted, or Manual calibration.",
        )

    def _replay(self, ok):
        return HealthCheckItem(
            "Replay",
            HealthStatus.PASS if ok else HealthStatus.FAIL,
            "Ready" if ok else "Replay buffer unavailable",
            "" if ok else "Restart live subsystem.",
        )

    def _audio(self, ok):
        return HealthCheckItem(
            "Audio",
            HealthStatus.PASS if ok else HealthStatus.WARN,
            "Ready" if ok else "Audio notification unavailable",
            "" if ok else "Check speaker volume/output device.",
            required=False,
        )

    def _tracking(self, status):
        status = str(status).upper()

        if status == "LOCKED":
            return HealthCheckItem(
                "Tracking",
                HealthStatus.PASS,
                "Locked",
            )

        if status == "SEARCHING":
            return HealthCheckItem(
                "Tracking",
                HealthStatus.WARN,
                "Searching",
                "Hold camera steady and keep the court visible.",
                required=False,
            )

        return HealthCheckItem(
            "Tracking",
            HealthStatus.FAIL,
            status or "LOST",
            "Reposition camera or recalibrate.",
        )

    def _fps(self, fps):
        fps = float(fps)

        if fps >= self.min_fps:
            return HealthCheckItem(
                "FPS",
                HealthStatus.PASS,
                f"{fps:.1f} fps",
            )

        return HealthCheckItem(
            "FPS",
            HealthStatus.FAIL,
            f"{fps:.1f} fps",
            "Lower resolution or use a faster camera/device.",
        )

    def _latency(self, latency_ms):
        latency_ms = float(latency_ms)

        if latency_ms <= self.max_latency_ms:
            return HealthCheckItem(
                "Latency",
                HealthStatus.PASS,
                f"{latency_ms:.1f} ms",
            )

        return HealthCheckItem(
            "Latency",
            HealthStatus.FAIL,
            f"{latency_ms:.1f} ms",
            "Reduce resolution or background load.",
        )

    def _storage(self, path):
        try:
            usage = shutil.disk_usage(Path(path).resolve())
            free_gb = usage.free / (1024 ** 3)
        except Exception:
            return HealthCheckItem(
                "Storage",
                HealthStatus.WARN,
                "Unable to determine free space",
                "Check available disk space manually.",
                required=False,
            )

        if free_gb >= self.min_free_gb:
            return HealthCheckItem(
                "Storage",
                HealthStatus.PASS,
                f"{free_gb:.1f} GB free",
            )

        return HealthCheckItem(
            "Storage",
            HealthStatus.FAIL,
            f"{free_gb:.1f} GB free",
            "Free disk space before starting the match.",
        )
