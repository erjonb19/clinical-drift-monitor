"""Stage a research job's inputs in the volume, pinned to the current commit.

Builds the cdm wheel from a clean HEAD, copies each named file as git stores it at HEAD into
``/Volumes/workspace/cdm/raw/deploy/<commit>/repo/``, uploads the notebook to the workspace,
and prints the wheel's SHA-256 and the ``files_sha256`` map for the job definition. The job's
notebook checks both before it reads anything.

Usage: python scripts/stage_research_job.py <notebook name> <repo file> [<repo file> ...]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHEEL = "cdm-0.1.0-py3-none-any.whl"
DEPLOY = "/Volumes/workspace/cdm/raw/deploy"


def run(*args: str, capture: bool = True) -> str:
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    out = subprocess.run(args, cwd=REPO, capture_output=capture, text=True, check=True, env=env)
    return out.stdout.strip() if capture else ""


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    notebook, files = sys.argv[1], sys.argv[2:]
    if run("git", "status", "--porcelain", "--", "src", "scripts", "config", "databricks", *files):
        sys.exit("uncommitted changes: commit first, so the staged files are the commit's")
    sha = run("git", "rev-parse", "--short", "HEAD")
    remote = f"{DEPLOY}/{sha}"
    with tempfile.TemporaryDirectory() as tmp:
        run(sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-q", "-w", tmp)
        wheel = Path(tmp) / WHEEL
        hashes = {}
        for rel in files:
            blob = subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True, check=True
            ).stdout
            hashes[rel] = hashlib.sha256(blob).hexdigest()
            local = Path(tmp) / "repo" / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(blob)
            run("databricks", "fs", "mkdirs", f"dbfs:{remote}/repo/{Path(rel).parent.as_posix()}")
            run("databricks", "fs", "cp", str(local), f"dbfs:{remote}/repo/{rel}", "--overwrite")
        run("databricks", "fs", "cp", str(wheel), f"dbfs:{remote}/{WHEEL}", "--overwrite")
        wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    user = json.loads(run("databricks", "current-user", "me", "-o", "json"))["userName"]
    source = REPO / "databricks" / "notebooks" / f"{notebook}.py"
    run("databricks", "workspace", "import", f"/Users/{user}/cdm/{notebook}", "--file",
        str(source), "--format", "SOURCE", "--language", "PYTHON", "--overwrite")  # fmt: skip
    print(json.dumps({"code_version": sha, "wheel": f"{remote}/{WHEEL}",
                      "wheel_sha256": wheel_sha, "repo": f"{remote}/repo",
                      "files_sha256": json.dumps(hashes)}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
