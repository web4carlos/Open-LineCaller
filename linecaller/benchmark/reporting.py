from __future__ import annotations

import json
from pathlib import Path


def write_report(report, root):
    root = Path(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    json_path = reports / "latest.json"
    txt_path = reports / "latest.txt"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )

    def fmt_optional(value, suffix=""):
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    text = "\n".join([
        "Open-LineCaller Benchmark Report",
        "================================",
        f"Clips:                    {report.clips}",
        f"Expected bounces:         {report.expected_bounces}",
        f"Detected bounces:         {report.detected_bounces}",
        f"Matched bounces:          {report.matched_bounces}",
        f"False positives:          {report.false_positive_bounces}",
        f"Missed bounces:           {report.missed_bounces}",
        f"Bounce precision:         {report.precision:.3f}",
        f"Bounce recall:            {report.recall:.3f}",
        f"Median frame error:       {fmt_optional(report.median_frame_error)}",
        f"Decision agreement:       {fmt_optional(report.decision_agreement)}",
        f"Calibration valid rate:   {report.calibration_valid_rate:.3f}",
    ])

    txt_path.write_text(text + "\n", encoding="utf-8")

    return json_path, txt_path
