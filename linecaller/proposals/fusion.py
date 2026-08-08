from __future__ import annotations

from dataclasses import replace
from collections import defaultdict

from .confidence import status_from_confidence
from .geometry import center_distance, iou
from .models import (
    BallProposal,
    ProposalResult,
    ProposalSource,
)


class ProposalFusion:
    def __init__(
        self,
        *,
        max_center_distance_px: float = 30.0,
        min_iou: float = 0.10,
        agreement_boost: float = 0.05,
    ):
        self.max_center_distance_px = float(max_center_distance_px)
        self.min_iou = float(min_iou)
        self.agreement_boost = float(agreement_boost)

    def fuse(
        self,
        frame_number: int,
        results: list[ProposalResult],
    ) -> ProposalResult:
        proposals = []

        for result in results:
            for p in result.proposals:
                pn = p.normalized()
                if not pn.validate():
                    proposals.append(pn)

        if not proposals:
            return ProposalResult(
                frame_number=frame_number,
                proposals=(),
                engine_name="ProposalFusion",
                latency_ms=sum(r.latency_ms for r in results),
            )

        proposals.sort(key=lambda p: p.confidence, reverse=True)

        clusters: list[list[BallProposal]] = []

        for proposal in proposals:
            placed = False

            for cluster in clusters:
                ref = cluster[0]

                if (
                    center_distance(proposal, ref) <= self.max_center_distance_px
                    or iou(proposal, ref) >= self.min_iou
                ):
                    cluster.append(proposal)
                    placed = True
                    break

            if not placed:
                clusters.append([proposal])

        fused = []

        for cluster in clusters:
            cluster.sort(key=lambda p: p.confidence, reverse=True)
            best = cluster[0]

            independent_sources = {
                p.source for p in cluster
            }

            confidence = best.confidence

            if len(independent_sources) > 1:
                confidence = min(
                    1.0,
                    confidence + self.agreement_boost,
                )

            metadata = dict(best.metadata)
            metadata.update({
                "fusion_cluster_size": len(cluster),
                "fusion_sources": sorted(s.value for s in independent_sources),
            })

            fused.append(
                replace(
                    best,
                    confidence=confidence,
                    source=(
                        ProposalSource.ENSEMBLE
                        if len(independent_sources) > 1
                        else best.source
                    ),
                    status=status_from_confidence(confidence),
                    metadata=metadata,
                )
            )

        fused.sort(key=lambda p: p.confidence, reverse=True)

        return ProposalResult(
            frame_number=frame_number,
            proposals=tuple(fused),
            engine_name="ProposalFusion",
            latency_ms=sum(r.latency_ms for r in results),
        )
