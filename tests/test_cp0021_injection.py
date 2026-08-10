from linecaller.live.pipeline_factory import create_live_pipeline_adapter


class CustomDetector:
    def detect(self, frame):
        return []


def test_custom_detector_remains_injectable():
    detector = CustomDetector()

    adapter = create_live_pipeline_adapter(detector=detector)

    assert adapter.perception.ball_engine.detector is detector
