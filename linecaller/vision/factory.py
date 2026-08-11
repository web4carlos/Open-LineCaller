from __future__ import annotations

from .providers.classical import ClassicalVisionProvider
from .providers.tracknet_csv import TrackNetCsvVisionProvider


def create_vision_provider(
    provider="classical",
    **kwargs,
):
    key = str(provider).strip().lower()

    if key == "classical":
        return ClassicalVisionProvider(**kwargs)

    if key in {"tracknet", "tracknet_csv"}:
        csv_path = kwargs.pop("csv_path", None)

        if csv_path is None:
            # Preserve CP-0022 Phase 1 contract:
            # TrackNet without an installed/evaluation artifact is still
            # considered unavailable.
            raise RuntimeError(
                "TrackNet provider requires a prediction CSV in "
                "CP-0022 Phase 2. Pass csv_path=..."
            )

        if kwargs:
            raise TypeError(
                f"Unsupported TrackNet options: {sorted(kwargs)}"
            )

        return TrackNetCsvVisionProvider(
            csv_path=csv_path
        )

    raise ValueError(
        f"Unknown vision provider: {provider}"
    )
