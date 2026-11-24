"""Backlog items 3 and 17: the challenger job (databricks/challenger_job.json) is wired so that one
GPU run trains, bundles and registers a challenger without touching the champion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
JOB = REPO / "databricks" / "challenger_job.json"


def job() -> dict[str, Any]:
    return json.loads(JOB.read_text(encoding="utf-8"))


def params(task: str) -> dict[str, str]:
    t = next(t for t in job()["tasks"] if t["task_key"] == task)
    return t["notebook_task"]["base_parameters"]


def widgets(notebook: str) -> set[str]:
    source = (REPO / "databricks" / "notebooks" / f"{notebook}.py").read_text(encoding="utf-8")
    block = source.split("}.items():")[0]
    return set(re.findall(r'^\s+"(\w+)":', block, flags=re.M))


def test_one_gpu_task_then_cpu_tasks_in_order_with_no_retries_or_schedule() -> None:
    spec = job()
    assert "schedule" not in spec and "trigger" not in spec
    assert spec["max_concurrent_runs"] == 1
    tasks = spec["tasks"]
    assert [t["task_key"] for t in tasks] == ["train_challenger", "serving_bundle", "register"]
    assert [t.get("depends_on") for t in tasks] == [
        None, [{"task_key": "train_challenger"}], [{"task_key": "serving_bundle"}],
    ]  # fmt: skip
    gpu = [t["task_key"] for t in tasks if t.get("compute", {}).get("hardware_accelerator")]
    assert gpu == ["train_challenger"]
    assert tasks[0]["timeout_seconds"] <= 3600
    assert all(t["max_retries"] == 0 and t["retry_on_timeout"] is False for t in tasks)


def test_every_parameter_is_a_widget_of_its_notebook() -> None:
    for task in job()["tasks"]:
        notebook = task["notebook_task"]["notebook_path"].rsplit("/", 1)[1]
        missing = set(task["notebook_task"]["base_parameters"]) - widgets(notebook)
        assert not missing, (task["task_key"], missing)


def test_the_tasks_hand_over_through_fixed_paths() -> None:
    train, bundle, register = (
        params("train_challenger"),
        params("serving_bundle"),
        params("register"),
    )
    assert train["keep_checkpoints"] == "true"
    assert bundle["models_file"] == f"{train['out']}/challenger_models.json"
    assert register["bundle"] == bundle["bundle_dir"]
    assert register["evaluated_results"] == f"{bundle['out']}/results.json"
    assert register["decision_path"] == f"{train['out']}/decision.json"
    staged = ("code_version", "wheel", "wheel_sha256", "repo", "files_sha256")
    assert len({tuple(p[k] for k in staged) for p in (train, bundle, register)}) == 1


def test_the_champion_is_never_touched() -> None:
    """The alias never moves (promote_challenger does that by hand), and nothing is written where
    the champion's results, bundle or the 2026-10-23 retraining record live."""
    train, bundle, register = (
        params("train_challenger"),
        params("serving_bundle"),
        params("register"),
    )
    assert register["alias"] == "" and register["model_gen"]
    champion = ("/phase3/retrain", "/phase3/serving", "/phase3/lineage", "/serving/d7191f4")
    for path in (train["out"], train["checkpoint_dir"], bundle["out"], bundle["bundle_dir"],
                 register["out"]):  # fmt: skip
        assert not any(path.endswith(c) or f"{c}/" in path for c in champion), path
    assert bundle["bundle_dir"].startswith("/Volumes/workspace/cdm/raw/deploy/serving/")
    assert train["checkpoint_dir"].startswith("/Volumes/workspace/cdm/raw/checkpoints/")


def test_the_spec_is_committed_and_names_the_rule_it_reuses() -> None:
    spec = json.loads((REPO / "results/phase3/challenger/spec.json").read_text(encoding="utf-8"))
    assert spec["committed_before_running"] is True
    assert "results/phase3/retrain/spec.json" in spec["rule"]
    assert params("train_challenger")["out"] in spec["outputs"]["decision"]
