from .providers.classical import ClassicalVisionProvider

def create_vision_provider(provider="classical", **kwargs):
    key = str(provider).strip().lower()
    if key == "classical":
        return ClassicalVisionProvider(**kwargs)
    if key == "tracknet":
        raise RuntimeError("TrackNet provider is not installed yet. CP-0022 Phase 1 only introduces the provider API.")
    raise ValueError(f"Unknown vision provider: {provider}")
