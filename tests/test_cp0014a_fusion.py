import pytest
from linecaller.proposals.fusion import ProposalFusion
from linecaller.proposals.models import (
    BallProposal,
    ProposalResult,
    ProposalSource,
)


def result(source, x, confidence):
    return ProposalResult(
        frame_number=10,
        engine_name=source.value,
        proposals=(
            BallProposal(
                frame_number=10,
                x=x,
                y=100,
                width=12,
                height=12,
                confidence=confidence,
                source=source,
            ),
        ),
    )


def test_fusion_merges_nearby_independent_sources():
    fused = ProposalFusion(
        max_center_distance_px=20,
        agreement_boost=.05,
    ).fuse(
        10,
        [
            result(ProposalSource.MOTION, 100, .80),
            result(ProposalSource.YOLO, 105, .90),
        ],
    )

    assert len(fused.proposals) == 1
    assert fused.proposals[0].source == ProposalSource.ENSEMBLE
    ##assert fused.proposals[0].confidence == .95
    assert fused.proposals[0].confidence == pytest.approx(.95)


def test_fusion_keeps_distant_candidates_separate():
    fused = ProposalFusion(
        max_center_distance_px=10,
        min_iou=.2,
    ).fuse(
        10,
        [
            result(ProposalSource.MOTION, 100, .8),
            result(ProposalSource.YOLO, 300, .9),
        ],
    )

    assert len(fused.proposals) == 2
