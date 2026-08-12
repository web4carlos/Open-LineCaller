def test_cp0026_imports():
    from linecaller.bounce_v2 import (
        BounceEngine,
        BounceEvent,
        MotionSample,
    )

    assert BounceEngine is not None
    assert BounceEvent is not None
    assert MotionSample is not None
