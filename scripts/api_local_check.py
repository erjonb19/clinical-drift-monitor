"""Checks of the API image run locally in Docker, committed so their numbers are traceable.

Starts the image, sends made-up images (plain green, plain grey and a 1600x1200 noise picture:
no patient data) and reads /health and the container's peak memory, then writes
``results/phase3/serving/api_local_checks.json``.

Usage: python scripts/api_local_check.py [--image cdm-api:local] [--port 7861]
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase3" / "serving" / "api_local_checks.json"
NAME = "cdm-api-local-check"
COLOURS = {"plain_green_400x300": (30, 160, 40), "plain_grey_400x300": (128, 128, 128)}


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def post(port: int, body: bytes) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://localhost:{port}/predict", data=body, headers={"Content-Type": "image/jpeg"}
    )
    with urllib.request.urlopen(request, timeout=60) as r:
        out: dict[str, object] = json.load(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="cdm-api:local")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    docker("run", "-d", "--name", NAME, "-p", f"{args.port}:7860", args.image)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2)
                break
            except OSError:
                time.sleep(2)
        made_up = {}
        for name, colour in COLOURS.items():
            buf = io.BytesIO()
            Image.new("RGB", (400, 300), colour).save(buf, format="JPEG")
            answer = post(args.port, buf.getvalue())
            made_up[name] = {k: answer[k] for k in ("label", "ood_score", "abstain")}
        # A full-size photograph's load: a made-up 1600x1200 random-noise image, three times.
        noise = Image.effect_noise((1600, 1200), 64).convert("RGB")
        buf = io.BytesIO()
        noise.save(buf, format="JPEG", quality=95)
        for _ in range(3):
            answer = post(args.port, buf.getvalue())
        made_up["noise_1600x1200"] = {k: answer[k] for k in ("label", "ood_score", "abstain")}
        with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=10) as r:
            health = json.load(r)
        peak = int(docker("exec", NAME, "sh", "-c", "cat /sys/fs/cgroup/memory.peak"))
        logs = docker("logs", NAME)
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    out = {
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "image": args.image,
        "image_id": docker("image", "inspect", "-f", "{{.Id}}", args.image),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),  # fmt: skip
        "made_up_images": made_up,
        "abstain_threshold": "from the bundle; see results/phase3/serving/results.json",
        "peak_memory_mib": round(peak / 2**20),
        "health": health,
        "access_log_lines": sum("POST /predict" in line for line in logs.splitlines()),
        "where": "the owner's laptop, Docker, no memory limit",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps({k: out[k] for k in ("made_up_images", "peak_memory_mib", "access_log_lines")})
    )


if __name__ == "__main__":
    main()
