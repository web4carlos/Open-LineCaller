import json

from linecaller.live.evidence_store import LiveEvidenceStore
from linecaller.live.models import LiveDecision, LiveEvidence


def test_evidence_store(tmp_path):
    path = tmp_path / "live.jsonl"

    store = LiveEvidenceStore(path)

    store.append(
        LiveEvidence(
            frame_number=10,
            decision=LiveDecision.IN,
            confidence=.99,
            replay_requested=False,
            decision_latency_ms=20,
            processing_fps=60,
        )
    )

    data = json.loads(path.read_text().strip())

    assert data["decision"] == "IN"
    assert data["frame_number"] == 10
