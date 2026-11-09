"""The API app for the serving benchmark (backlog item 13): the same create_app as
app/api/main.py, with the rate limiter opened wide so the benchmark measures serving, not the
limiter. Never deployed.

Usage: BUNDLE_DIR=<bundle> python scripts/bench_app.py [--port 7861]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from app.api.main import RateLimiter, create_app
from cdm.serving import Service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    bundle = Path(os.environ["BUNDLE_DIR"])
    limiter = RateLimiter(per_client=10**9, total=10**9)
    app = create_app(lambda: Service.load(bundle), limiter)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
