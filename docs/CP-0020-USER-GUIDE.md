# CP-0020 User Guide

## Install

Run from the extracted pack:

    .\install_into_repo.bat

## Confirm real perception is wired

From C:\olinecaller:

    .\.venv\Scripts\python.exe -c "from linecaller.live.pipeline_factory import create_live_pipeline_adapter; a=create_live_pipeline_adapter(); print(type(a).__name__); print(type(a.perception).__name__)"

Expected:

    RealTimeOfficiatingAdapter
    RealArtificialVisionPerceptionAdapter

It must NOT print NullLivePerceptionAdapter.

## Test Smart Annotation Studio again

    .\.venv\Scripts\python.exe -m tools.smart_annotation_studio

Open:

    C:\validation\match001_1080p60.mp4

Press:

    Scan Candidates

Without calibration, real bounce candidates appear as REVIEW candidates.
The operator still labels them IN / OUT / SKIP.

## Live

The same factory is used by the live product path. CP-0020 therefore removes
the NULL perception default for both validation and live operation.
