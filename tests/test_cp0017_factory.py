from linecaller.live.pipeline_factory import create_live_pipeline_adapter
from linecaller.live.realtime_adapter import RealTimeOfficiatingAdapter


def test_factory_returns_adapter():
    adapter = create_live_pipeline_adapter()
    assert isinstance(adapter, RealTimeOfficiatingAdapter)
