# CP-0021 User Guide

Before install:

    cd C:\olinecaller
    git status

Install only when clean:

    .\install_into_repo.bat

Verify detector:

    .\.venv\Scripts\python.exe -c "from linecaller.live.pipeline_factory import create_live_pipeline_adapter; a=create_live_pipeline_adapter(); print(type(a.perception.ball_engine.detector).__name__)"

Expected:

    AdvancedMotionBallDetector

Optional quick probe:

    .\.venv\Scripts\python.exe -m tools.detector_probe `
      --video C:\validation\match001_1080p60.mp4 `
      --max-frames 1800

Then re-run Smart Annotation Studio and compare against the CP-0020 baseline of
1 candidate event.
