"""Serving benchmark for the API (backlog item 13): latency and throughput at several numbers of
concurrent users, and the server's memory.

Starts scripts/bench_app.py with the given bundle, sends one fixed synthetic JPEG from N
concurrent users (threads, standard library HTTP), and writes the numbers with the hardware,
the software versions and the bundle version. With the CI test bundle (untrained weights) the
numbers are **serving speed only, never accuracy**: the model's answers mean nothing.

Usage: python scripts/load_test.py --bundle dist/test_bundle [--users 1 5 20]
       [--requests 200] [--out results/phase3/serving/benchmark.json]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
UNTRAINED = (
    "the CI test bundle has untrained weights: these numbers measure serving speed only, "
    "never accuracy"
)
SERVER = (
    "scripts/bench_app.py: app/api/main.py's create_app, one uvicorn worker, "
    "rate limiter opened wide"
)


def image() -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(60, 200, (448, 448, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def post(url: str, body: bytes) -> tuple[int, float]:
    start = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "image/jpeg"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as err:
        code = err.code
    return code, 1000 * (time.perf_counter() - start)


def memory_kb(pid: int) -> dict[str, int]:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[key] = int(value.split()[0])
    return out


def level(url: str, body: bytes, users: int, total: int) -> dict[str, Any]:
    per = max(1, total // users)
    lat: list[float] = []
    codes: dict[int, int] = {}
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(per):
            code, ms = post(url, body)
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if code == 200:
                    lat.append(ms)

    threads = [threading.Thread(target=worker) for _ in range(users)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    if not lat:
        raise RuntimeError(f"no request succeeded at {users} users: {codes}")
    q = [float(x) for x in np.percentile(lat, [50, 95, 99])]
    return {
        "users": users,
        "requests": users * per,
        "status_codes": codes,
        "latency_ms": {
            "p50": round(float(q[0]), 1),
            "p95": round(float(q[1]), 1),
            "p99": round(float(q[2]), 1),
            "mean": round(statistics.fmean(lat), 1),
        },
        "throughput_per_s": round(len(lat) / wall, 2),
        "wall_s": round(wall, 1),
    }


def hardware() -> dict[str, Any]:
    cpu = "unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem = int(Path("/proc/meminfo").read_text().split()[1]) // 1024
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_mb": mem, "os": platform.platform()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--where", default="unspecified machine")
    parser.add_argument("--out", type=Path, default=REPO / "results/phase3/serving/benchmark.json")
    args = parser.parse_args()
    import torch

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    env = {**os.environ, "BUNDLE_DIR": str(args.bundle), "PYTHONPATH": str(REPO)}
    server = subprocess.Popen(
        [sys.executable, str(REPO / "scripts/bench_app.py"), "--port", str(args.port)], env=env
    )
    url = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/health", timeout=5).read()
                break
            except OSError:
                time.sleep(1)
        body = image()
        for _ in range(10):  # warm-up, not counted
            post(url + "/predict", body)
        idle = memory_kb(server.pid)
        levels = []
        for users in args.users:
            levels.append(level(url + "/predict", body, users, args.requests))
            levels[-1]["server_memory_kb"] = memory_kb(server.pid)
    finally:
        server.terminate()
        server.wait(timeout=30)
    result = {
        "what": "serving speed of the API (backlog item 13); not accuracy",
        "bundle_version": manifest["version"],
        "caveat": UNTRAINED if "untrained" in manifest["version"] else "",
        "measured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "uncommitted_changes": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "scripts", "app", "src"],
                cwd=REPO,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "where": args.where,
        "hardware": hardware(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "server": SERVER,
        "image": "one synthetic 448x448 JPEG, sent repeatedly; 10 warm-up requests not counted",
        "memory_after_warmup_kb": idle,
        "levels": levels,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for lv in levels:
        print(lv["users"], lv["latency_ms"], lv["throughput_per_s"], lv["server_memory_kb"])


if __name__ == "__main__":
    main()
