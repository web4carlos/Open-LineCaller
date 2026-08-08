from linecaller.live.pipeline_models import LivePipelineResult


def test_pipeline_result_defaults():
    result = LivePipelineResult(frame_number=1)

    assert result.tracking_status == "SEARCHING"
    assert result.decision is None
    assert result.bounce_detected is False
