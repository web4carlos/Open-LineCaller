def test_product_shell_imports():
    from linecaller.product_shell import OpenLineCallerMainWindow
    assert OpenLineCallerMainWindow is not None

def test_live_factory_imports():
    from linecaller.live.pipeline_factory import create_live_pipeline_adapter
    assert callable(create_live_pipeline_adapter)
