from pathlib import Path
import csv
import pytest

from linecaller.live_bridge.officiating_csv import CsvOfficiatingTimeline


FIELDS = [
    "frame","image_x","image_y","nearest_line","geometry_state",
    "bounce_confidence","bounce_score","decision_confidence",
    "signed_distance_in","gate_decision","gate_confidence",
    "force_review","final_decision","officiating_reason",
]


def row(**overrides):
    x = {
        "frame":"29",
        "image_x":"885.585",
        "image_y":"485.117",
        "nearest_line":"FAR_BASELINE",
        "geometry_state":"OUTSIDE",
        "bounce_confidence":"0.556939",
        "bounce_score":"0.626863",
        "decision_confidence":"0.556939",
        "signed_distance_in":"-61.442",
        "gate_decision":"VERIFIED_BOUNCE",
        "gate_confidence":"0.5663",
        "force_review":"False",
        "final_decision":"OUT",
        "officiating_reason":"VERIFIED_BOUNCE_BASE_DECISION_PRESERVED",
    }
    x.update(overrides)
    return x


def write(tmp_path, rows, fields=FIELDS):
    p = tmp_path / "final_officiating_decisions.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def test_loads_validated_final_csv(tmp_path):
    t=CsvOfficiatingTimeline.load(write(tmp_path,[row()]))
    assert t.events[0].final_decision=="OUT"


def test_preserves_frame_and_pixel_location(tmp_path):
    e=CsvOfficiatingTimeline.load(write(tmp_path,[row()])).events[0]
    assert e.frame==29 and e.image_x==885.585 and e.image_y==485.117


def test_preserves_signed_distance_inches(tmp_path):
    e=CsvOfficiatingTimeline.load(write(tmp_path,[row()])).events[0]
    assert e.signed_distance_in==-61.442


def test_preserves_gate_metadata(tmp_path):
    e=CsvOfficiatingTimeline.load(write(tmp_path,[row()])).events[0]
    assert e.gate_decision=="VERIFIED_BOUNCE"
    assert e.force_review is False


def test_review_is_supported(tmp_path):
    e=CsvOfficiatingTimeline.load(write(tmp_path,[row(
        frame="94",
        final_decision="REVIEW",
        force_review="True",
        gate_decision="REVIEW_BOUNCE",
    )])).events[0]
    assert e.final_decision=="REVIEW"
    assert e.force_review is True


def test_force_review_cannot_publish_out(tmp_path):
    with pytest.raises(ValueError):
        CsvOfficiatingTimeline.load(write(tmp_path,[row(
            force_review="True",
            final_decision="OUT",
        )]))


def test_invalid_final_decision_rejected(tmp_path):
    with pytest.raises(ValueError):
        CsvOfficiatingTimeline.load(write(tmp_path,[row(final_decision="MAYBE")]))


def test_missing_required_fields_rejected(tmp_path):
    fields=[x for x in FIELDS if x!="final_decision"]
    with pytest.raises(ValueError):
        CsvOfficiatingTimeline.load(write(tmp_path,[],fields=fields))


def test_events_sorted_by_frame(tmp_path):
    t=CsvOfficiatingTimeline.load(write(tmp_path,[row(frame="157"),row(frame="29")]))
    assert [e.frame for e in t.events]==[29,157]


def test_react_uses_final_decision():
    s=Path("ui/src/LiveCourtVideo.tsx").read_text(encoding="utf-8")
    assert "final_decision" in s
    assert "FINAL OFFICIATING EVENTS" in s
