from linecaller.proposals.factory import create_default_proposal_engine
from linecaller.proposals.yolo_engine import YOLOProposalEngine


def test_factory_uses_environment(monkeypatch):
    monkeypatch.setenv(
        "OPEN_LINECALLER_YOLO_WEIGHTS",
        "custom.pt",
    )
    monkeypatch.setenv(
        "OPEN_LINECALLER_YOLO_CLASS",
        "pickleball",
    )

    engine = create_default_proposal_engine()

    assert isinstance(engine, YOLOProposalEngine)
    assert engine.weights == "custom.pt"
    assert engine.class_name == "pickleball"
