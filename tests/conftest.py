from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


# Keep every pytest run away from the operator's real .linecaller_runtime.
_TEST_RUNTIME = (
    Path(tempfile.gettempdir())
    / f"open-linecaller-pytest-{os.getpid()}"
)
os.environ["LINECALLER_RUNTIME_DIR"] = str(_TEST_RUNTIME)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_RUNTIME, ignore_errors=True)
