from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import median

from linecaller.benchmark.manifest import load_manifest
from linecaller.benchmark.matching import match_events
from linecaller.benchmark.metrics import aggregate, safe_ratio
from linecaller.benchmark.models import ClipBenchmarkResult
from linecaller.calibration.profile import CalibrationProfile
from linecaller.calibration.quality import CalibrationStatus
from linecaller.pipeline.integrated import IntegratedVideoPipeline


class BenchmarkRunner:
    def __init__(
        self,
        *,
        frame_tolerance: int = 2,
        pipeline_factory=None,
    ):
        self.frame_tolerance = int(frame_tolerance)
        self.pipeline_factory = pipeline_factory or IntegratedVideoPipeline

    @staticmethod
    def _read_expected(path: Path) -> list[dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("bounces", []))

    @staticmethod
    def _read_detected_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []

        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def run_clip(self, clip):
        expected = self._read_expected(clip.expected)

        calibration = CalibrationProfile.load(clip.calibration)
        calibration_valid = calibration.status == CalibrationStatus.VALID

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output_video = tmp / "annotated.mp4"
            events_path = tmp / "events.jsonl"

            pipeline = self.pipeline_factory()

            pipeline.run(
                video_path=clip.video,
                calibration_path=clip.calibration,
                output_video_path=output_video,
                events_path=events_path,
            )

            detected = self._read_detected_jsonl(events_path)

        matches, fp, fn = match_events(
            expected,
            detected,
            frame_tolerance=self.frame_tolerance,
        )

        matched = len(matches)
        precision = safe_ratio(matched, matched + fp)
        recall = safe_ratio(matched, matched + fn)

        errors = [m.frame_error for m in matches]
        decisions = [
            m.decision_match
            for m in matches
            if m.decision_match is not None
        ]

        agreement = None
        if decisions:
            agreement = sum(1 for x in decisions if x) / len(decisions)

        return ClipBenchmarkResult(
            clip_id=clip.clip_id,
            expected_bounces=len(expected),
            detected_bounces=len(detected),
            matched_bounces=matched,
            false_positive_bounces=fp,
            missed_bounces=fn,
            precision=precision,
            recall=recall,
            median_frame_error=median(errors) if errors else None,
            decision_agreement=agreement,
            calibration_valid=calibration_valid,
            matches=matches,
        )

    def run(self, root):
        clips = load_manifest(root)
        results = [self.run_clip(clip) for clip in clips]
        return aggregate(results)
