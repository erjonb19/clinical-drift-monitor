"""Deploy Phase 1 to Databricks Free Edition with the Databricks CLI (no bundles).

Run from the repository root on a clean commit, with the CLI installed and logged in
(docs/phase1-runbook.md). It:

1. refuses to deploy uncommitted code, and pins everything to the commit hash;
2. creates the schemas and the raw volume if missing;
3. uploads the cdm wheel, config/sites.json (with splits_v1.csv, patient_overlap_v1.csv,
   omop_concepts.csv and the private resolved OMOP concepts beside it),
   HAM10000_metadata.csv (MD5-checked) and the ImageNet weights (hash-checked) into the
   volume, and the pipeline source into the workspace;
4. creates or updates the pipelines ``cdm-phase1`` and ``cdm-phase1-broken`` and the jobs
   ``cdm-phase1`` and ``cdm-phase1-broken-demo`` from databricks/*.json.

``--job databricks/<name>_job.json`` only creates or updates that one job. The committed job
files hold ``{{DATABRICKS_USER}}`` where the workspace user name goes (no email in git); it is
filled with the logged-in user.

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

import pandas as pd

from cdm.data import data_root
from cdm.gold import WEIGHTS_FILE
from cdm.omop import load_concept_map, notes_quoting_concepts

REPO = Path(__file__).resolve().parents[1]
CODE = ["src", "pipelines", "config", "pyproject.toml"]
WHEEL = "cdm-0.1.0-py3-none-any.whl"
# Made by scripts/omop_concepts.py from a licensed Athena download; SNOMED content, never in git.
RESOLVED_CONCEPTS = Path.home() / "athena_vocab" / "cdm_resolved_concepts.csv"


def check_resolved(cmap_path: Path, resolved: Path) -> None:
    """The resolved concepts must be exactly the map's non-zero concepts: a stale file (map
    changed, scripts/omop_concepts.py not rerun) would fail the omop_concepts_resolve gate
    only after a full pipeline run."""
    if not resolved.exists():
        sys.exit(f"{resolved} missing: run scripts/omop_concepts.py --vocab <Athena folder>")
    wanted = {int(c) for c in load_concept_map(cmap_path)["concept_id"] if c != 0}
    have = set(pd.read_csv(resolved, dtype=str)["concept_id"].astype(int))
    if wanted != have:
        sys.exit(f"{resolved} is stale ({len(have ^ wanted)} concepts differ): rerun omop_concepts")
    quoting = notes_quoting_concepts(load_concept_map(cmap_path), pd.read_csv(resolved, dtype=str))
    if quoting:
        problems = "; ".join(quoting)
        sys.exit(f"concept map notes quote SNOMED names, which must not be published: {problems}")


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


def current_user(cli: Cli, dry_run: bool) -> str:
    if dry_run:
        return "DRY-RUN-USER"
    name: str = json.loads(cli("current-user", "me", "--output", "json"))["userName"]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Databricks CLI profile")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", type=Path, help="only create or update this job definition")
    args = parser.parse_args()

    if args.job:
        cli = Cli(args.profile, args.dry_run)
        user = current_user(cli, args.dry_run)
        spec = render(args.job.read_text(encoding="utf-8"), {"DATABRICKS_USER": user})
        (REPO / "dist").mkdir(exist_ok=True)
        print(f"job {spec['name']} = {upsert_job(cli, spec, args.dry_run)}")
        return 0

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
    check_resolved(REPO / "config" / "omop_concepts.csv", RESOLVED_CONCEPTS)
    weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / WEIGHTS_FILE
    if not weights.exists():
        sys.exit(f"{weights} missing: run any Phase 0 command once to download it")

    cli = Cli(args.profile, args.dry_run)
    me = current_user(cli, args.dry_run)
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
        REPO / "config" / "splits_v1.csv": f"{deploy}/splits_v1.csv",
        REPO / "config" / "patient_overlap_v1.csv": f"{deploy}/patient_overlap_v1.csv",
        REPO / "config" / "omop_concepts.csv": f"{deploy}/omop_concepts.csv",
        RESOLVED_CONCEPTS: f"{deploy}/cdm_resolved_concepts.csv",
        ham_csv: f"{raw}/uploads/{ham_csv.name}",
        weights: f"{raw}/deploy/weights/{WEIGHTS_FILE}",
    }
    for local, remote in uploads.items():
        # fs cp does not create parent folders inside a volume.
        cli("fs", "mkdirs", f"dbfs:{remote.rsplit('/', 1)[0]}")
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
    # The broken run reads only the small faulted subset that `cdm-ingest --make-broken`
    # copies into broken/, with that subset's own expected fingerprint and class counts.
    broken = {
        "PIPELINE_NAME": "cdm-phase1-broken",
        "SCHEMA": f"{schema}_broken",
        "CONFIG": f"{raw}/broken/sites.json",
        "IMAGES_GLOB": f"{raw}/broken/images/*/*.jpg",
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
