from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Running a script inside tools/ makes Python use tools/ as sys.path[0].
# Insert the repository root so "linecaller" is importable from the documented
# PowerShell command: python tools\run_live_api.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Open LineCaller CP-0036 live FastAPI bridge."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn development reload.",
    )
    args = parser.parse_args()

    uvicorn.run(
        "linecaller.api.live_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()