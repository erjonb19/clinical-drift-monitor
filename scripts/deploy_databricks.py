"""Deploy Phase 1 to Databricks Free Edition with the Databricks CLI (no bundles).

Run from the repository root on a clean commit, with the CLI installed and logged in
(docs/phase1-runbook.md). It:

1. refuses to deploy uncommitted code, and pins everything to the commit hash;
2. creates the schemas and the raw volume if missing;
3. uploads the cdm wheel, config/sites.json, HAM10000_metadata.csv (MD5-checked) and the
   ImageNet weights (hash-checked) into the volume, and the pipeline source into the workspace;
4. creates or updates the pipelines ``cdm-phase1`` and ``cdm-phase1-broken`` and the jobs
   ``cdm-phase1`` and ``cdm-phase1-broken-demo`` from databricks/*.json.

``--dry-run`` prints every command and rendered definition without running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from cdm.data import data_root
from cdm.gold import WEIGHTS_FILE

REPO = Path(__file__).resolve().parents[1]
CODE = ["src", "pipelines", "config", "pyproject.toml"]
WHEEL = "cdm-0.1.0-py3-none-any.whl"


def render(template: str, values: dict[str, str]) -> dict[str, Any]:
    """Fill ``{{NAME}}`` placeholders; refuse to return a definition with any left over."""
    text = template
    for name, value in values.items():
        text = text.replace("{{" + name + "}}", value)
    left = sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", text)))
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    rendered: dict[str, Any] = json.loads(text)
    return rendered


class Cli:
    def __init__(self, profile: str | None, dry_run: bool) -> None:
        self.base = ["databricks", *(["--profile", profile] if profile else [])]
        self.dry_run = dry_run

    def __call__(self, *args: str, ok_if: str | None = None) -> str:
        cmd = [*self.base, *args]
        print("$", " ".join(cmd), flush=True)
        if self.dry_run:
            return "{}"
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            if ok_if and ok_if in (done.stderr + done.stdout):
                return ""
            sys.exit(f"command failed ({done.returncode}):\n{done.stderr or done.stdout}")
        return done.stdout


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def upsert_pipeline(cli: Cli, spec: dict[str, Any], dry_run: bool) -> str:
    listing = json.loads(cli("pipelines", "list-pipelines", "--output", "json") or "[]")
    existing = (
        [p for p in listing if p.get("name") == spec["name"]] if isinstance(listing, list) else []
    )
    path = REPO / "dist" / f"{spec['name']}.pipeline.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    if existing:
        pid = str(existing[0]["pipeline_id"])
        cli("pipelines", "update", pid, "--json", f"@{path}")
        return pid
    out = cli("pipelines", "create", "--json", f"@{path}")
    return "DRY-RUN-PIPELINE-ID" if dry_run else str(json.loads(out)["pipeline_id"])


def upsert_job(cli: Cli, spec: dict[str, Any], dry_run: bool) -> str:
    listing = json.loads(cli("jobs", "list", "--name", spec["name"], "--output", "json") or "[]")
    jobs = listing if isinstance(listing, list) else listing.get("jobs", [])
    path = REPO / "dist" / f"{spec['name']}.job.json"
    if jobs:
        job_id = str(jobs[0]["job_id"])
        path.write_text(
            json.dumps({"job_id": int(job_id), "new_settings": spec}, indent=2), encoding="utf-8"
        )
        cli("jobs", "reset", "--json", f"@{path}")
        return job_id
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    out = cli("jobs", "create", "--json", f"@{path}")
    return "DRY-RUN-JOB-ID" if dry_run else str(json.loads(out)["job_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Databricks CLI profile")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if git("status", "--porcelain", "--", *CODE):
        sys.exit("uncommitted changes in src/, pipelines/, config/ or pyproject.toml: commit first")
    sha = git("rev-parse", "--short", "HEAD")
    config = json.loads((REPO / "config" / "sites.json").read_text(encoding="utf-8"))
    catalog, schema, volume = config["catalog"], config["schema"], config["volume"]
    raw = f"/Volumes/{catalog}/{schema}/{volume}"
    deploy = f"{raw}/deploy/{sha}"

    ham_csv = data_root() / "ham10000" / config["ham10000"]["metadata_file"]
    if hashlib.md5(ham_csv.read_bytes()).hexdigest() != config["ham10000"]["metadata_md5"]:
        sys.exit(f"{ham_csv} does not match the MD5 in config/sites.json")
    weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / WEIGHTS_FILE
    if not weights.exists():
        sys.exit(f"{weights} missing: run any Phase 0 command once to download it")

    cli = Cli(args.profile, args.dry_run)
    me = (
        "DRY-RUN-USER"
        if args.dry_run
        else json.loads(cli("current-user", "me", "--output", "json"))["userName"]
    )
    for s in (schema, f"{schema}_broken"):
        cli("schemas", "create", s, catalog, ok_if="already exists")
    cli("volumes", "create", catalog, schema, volume, "MANAGED", ok_if="already exists")

    (REPO / "dist").mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-q", "-w", "dist"],
        cwd=REPO,
        check=True,
    )
    uploads = {
        REPO / "dist" / WHEEL: f"{deploy}/{WHEEL}",
        REPO / "config" / "sites.json": f"{deploy}/sites.json",
        ham_csv: f"{raw}/uploads/{ham_csv.name}",
        weights: f"{raw}/deploy/weights/{WEIGHTS_FILE}",
    }
    for local, remote in uploads.items():
        cli("fs", "cp", str(local), f"dbfs:{remote}", "--overwrite")
    workspace_dir = f"/Users/{me}/cdm/{sha}"
    cli("workspace", "mkdirs", workspace_dir)
    pipeline_file = f"/Workspace{workspace_dir}/lakehouse.py"
    source = str(REPO / "pipelines" / "lakehouse.py")
    target = f"{workspace_dir}/lakehouse.py"
    cli("workspace", "import", target, "--file", source, "--format", "RAW", "--overwrite")

    def template(name: str) -> str:
        return (REPO / "databricks" / name).read_text(encoding="utf-8")

    common = {
        "SHA": sha,
        "CATALOG": catalog,
        "WHEEL": f"{deploy}/{WHEEL}",
        "CONFIG": f"{deploy}/sites.json",
        "PIPELINE_FILE": pipeline_file,
        "RAW": raw,
        "WEIGHTS": f"{raw}/deploy/weights/{WEIGHTS_FILE}",
    }
    real = {
        "PIPELINE_NAME": "cdm-phase1",
        "SCHEMA": schema,
        "IMAGES_GLOB": f"{raw}/images/*/*.jpg",
        "METADATA_PATH": f"{raw}/metadata",
        "MANIFEST_PATH": f"{raw}/manifest",
    }
    # The broken run reads the real images plus the faulted extras, and the faulted records.
    broken = {
        "PIPELINE_NAME": "cdm-phase1-broken",
        "SCHEMA": f"{schema}_broken",
        "IMAGES_GLOB": f"{raw}/{{images,broken/images}}/*/*.jpg",
        "METADATA_PATH": f"{raw}/broken/metadata",
        "MANIFEST_PATH": f"{raw}/broken/manifest",
    }
    pipeline_id = upsert_pipeline(
        cli, render(template("pipeline.json"), {**common, **real}), args.dry_run
    )
    broken_id = upsert_pipeline(
        cli, render(template("pipeline.json"), {**common, **broken}), args.dry_run
    )
    ids = {**common, "SCHEMA": schema, "PIPELINE_ID": pipeline_id, "BROKEN_PIPELINE_ID": broken_id}
    job = upsert_job(cli, render(template("job.json"), ids), args.dry_run)
    demo = upsert_job(cli, render(template("job_broken.json"), ids), args.dry_run)

    print(f"\ndeployed {sha}: job cdm-phase1 = {job}, job cdm-phase1-broken-demo = {demo}")
    print(f"run:  databricks jobs run-now {job}")
    print(f"then: databricks jobs run-now {demo}   (must fail at the pipeline's gate_results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
