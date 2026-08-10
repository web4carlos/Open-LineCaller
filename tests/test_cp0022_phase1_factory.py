import pytest
from linecaller.vision.factory import create_vision_provider
from linecaller.vision.providers.classical import ClassicalVisionProvider

def test_factory_classical():
    assert isinstance(create_vision_provider("classical"), ClassicalVisionProvider)

def test_factory_unknown():
    with pytest.raises(ValueError):
        create_vision_provider("nope")

def test_tracknet_placeholder():
    with pytest.raises(RuntimeError):
        create_vision_provider("tracknet")
