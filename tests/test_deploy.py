"""The committed Databricks definitions render completely and point where they should."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def deploy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deploy", REPO / "scripts" / "deploy_databricks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALUES = {
    "SHA": "abc1234",
    "CATALOG": "workspace",
    "SCHEMA": "cdm",
    "WHEEL": "/Volumes/workspace/cdm/raw/deploy/abc1234/cdm-0.1.0-py3-none-any.whl",
    "CONFIG": "/Volumes/workspace/cdm/raw/deploy/abc1234/sites.json",
    "PIPELINE_FILE": "/Workspace/Users/me/cdm/abc1234/lakehouse.py",
    "PIPELINE_NAME": "cdm-phase1-broken",
    "RAW": "/Volumes/workspace/cdm/raw",
    "WEIGHTS": "/Volumes/workspace/cdm/raw/deploy/weights/w.pth",
    "IMAGES_GLOB": "/Volumes/workspace/cdm/raw/{images,broken/images}/*/*.jpg",
    "METADATA_PATH": "/Volumes/workspace/cdm/raw/broken/metadata",
    "MANIFEST_PATH": "/Volumes/workspace/cdm/raw/broken/manifest",
    "PIPELINE_ID": "p-1",
    "BROKEN_PIPELINE_ID": "p-2",
}


@pytest.mark.parametrize("name", ["pipeline.json", "job.json", "job_broken.json"])
def test_every_definition_renders_with_nothing_left_over(name: str) -> None:
    template = (REPO / "databricks" / name).read_text(encoding="utf-8")
    spec = deploy_module().render(template, VALUES)
    assert "{{" not in str(spec)


def test_pipeline_is_serverless_and_reads_both_image_folders_for_the_broken_demo() -> None:
    template = (REPO / "databricks" / "pipeline.json").read_text(encoding="utf-8")
    spec = deploy_module().render(template, VALUES)
    assert spec["serverless"] is True and spec["continuous"] is False
    assert spec["configuration"]["cdm.images_glob"].endswith("{images,broken/images}/*/*.jpg")
    assert spec["environment"]["dependencies"] == [VALUES["WHEEL"]]


def test_job_runs_ingest_then_pipeline_then_gold() -> None:
    template = (REPO / "databricks" / "job.json").read_text(encoding="utf-8")
    tasks = deploy_module().render(template, VALUES)["tasks"]
    order = {t["task_key"]: [d["task_key"] for d in t.get("depends_on", [])] for t in tasks}
    assert order == {"ingest": [], "pipeline": ["ingest"], "gold": ["pipeline"]}


def test_a_missing_value_is_refused() -> None:
    template = (REPO / "databricks" / "job.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="PIPELINE_ID"):
        deploy_module().render(template, {k: v for k, v in VALUES.items() if k != "PIPELINE_ID"})
