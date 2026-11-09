"""The committed Databricks definitions render completely and point where they should."""

from __future__ import annotations

import importlib.util
import json
import re
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
    "IMAGES_GLOB": "/Volumes/workspace/cdm/raw/broken/images/*/*.jpg",
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


def test_broken_pipeline_is_serverless_and_reads_only_the_broken_subset() -> None:
    template = (REPO / "databricks" / "pipeline.json").read_text(encoding="utf-8")
    spec = deploy_module().render(template, VALUES)
    assert spec["serverless"] is True and spec["continuous"] is False
    assert spec["configuration"]["cdm.images_glob"].endswith("/raw/broken/images/*/*.jpg")
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


def test_broken_demo_job_fully_refreshes_its_pipeline() -> None:
    template = (REPO / "databricks" / "job_broken.json").read_text(encoding="utf-8")
    tasks = {t["task_key"]: t for t in deploy_module().render(template, VALUES)["tasks"]}
    assert tasks["pipeline"]["pipeline_task"]["full_refresh"] is True


def test_pipelines_do_not_retry_failed_updates() -> None:
    """A failed gate is deterministic: retrying repeats the failure and spends quota. The
    broken demo retried its update 5 times before this was set (run 561727450716511)."""
    template = (REPO / "databricks" / "pipeline.json").read_text(encoding="utf-8")
    spec = deploy_module().render(template, VALUES)
    assert spec["configuration"]["pipelines.numUpdateRetryAttempts"] == "0"


def test_deploy_refuses_missing_or_stale_resolved_concepts(tmp_path: Path) -> None:
    check = deploy_module().check_resolved
    cmap = tmp_path / "map.csv"
    cmap.write_text(
        "source_field,scope,source_value,concept_id,expected_domain,match,note\n"
        "label,all,nv,4271013,Condition,exact,\nlabel,all,x,0,Condition,none,why\n",
        encoding="utf-8",
    )
    resolved = tmp_path / "resolved.csv"
    with pytest.raises(SystemExit, match="missing"):
        check(cmap, resolved)
    head = "concept_id,concept_name,vocabulary_id\n"
    resolved.write_text(head + "4271013,n,SNOMED\n1,stale,SNOMED\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        check(cmap, resolved)
    resolved.write_text(head + "4271013,melanocytic naevus,SNOMED\n", encoding="utf-8")
    check(cmap, resolved)
    cmap.write_text(cmap.read_text("utf-8").replace(",why", ",a melanocytic naevus"), "utf-8")
    with pytest.raises(SystemExit, match="quote SNOMED names"):
        check(cmap, resolved)


def test_no_committed_definition_holds_an_email_address() -> None:
    """The workspace user name is an email address; the job files carry a placeholder."""
    for path in (REPO / "databricks").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), path.name


@pytest.mark.parametrize(
    "path", sorted((REPO / "databricks").glob("*_job.json")), ids=lambda p: p.name
)
def test_recorded_jobs_render_with_only_the_user(path: Path) -> None:
    spec = deploy_module().render(path.read_text(encoding="utf-8"), {"DATABRICKS_USER": "u@x.org"})
    paths = [t["notebook_task"]["notebook_path"] for t in spec["tasks"] if "notebook_task" in t]
    assert paths and all(p.startswith("/Users/u@x.org/cdm/") for p in paths)


def test_job_mode_creates_one_recorded_job_without_an_email(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = deploy_module()
    path = REPO / "databricks" / "drift_job.json"
    monkeypatch.setattr("sys.argv", ["deploy", "--job", str(path), "--dry-run"])
    assert module.main() == 0
    name = json.loads(path.read_text(encoding="utf-8"))["name"]
    assert f"job {name} = DRY-RUN-JOB-ID" in capsys.readouterr().out
    written = (REPO / "dist" / f"{name}.job.json").read_text(encoding="utf-8")
    assert "DRY-RUN-USER" in written and "@" not in written


def test_job_mode_refuses_the_deploy_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["deploy", "--job", str(REPO / "databricks" / "job.json")])
    with pytest.raises(SystemExit, match="full run"):
        deploy_module().main()


@pytest.mark.parametrize(
    "path", sorted((REPO / "databricks").glob("*_job.json")), ids=lambda p: p.name
)
def test_every_scheduled_job_is_unpaused_and_in_utc(path: Path) -> None:
    """Backlog item 1: a committed schedule is never paused, and every time is UTC."""
    schedule = json.loads(path.read_text(encoding="utf-8")).get("schedule")
    if schedule is not None:
        assert schedule["pause_status"] == "UNPAUSED", path.name
        assert schedule["timezone_id"] == "UTC", path.name


def test_the_scheduler_probe_is_small_and_writes_only_its_own_folder() -> None:
    """Backlog item 1's probe: the monthly job's kind of schedule, firing on 8 to 10 November at
    06:00 UTC; one file per run in its own folder; no table, no wheel, a short timeout."""
    spec = json.loads((REPO / "databricks" / "scheduler_probe_job.json").read_text("utf-8"))
    monthly = json.loads((REPO / "databricks" / "monthly_drift_job.json").read_text("utf-8"))
    assert spec["schedule"]["quartz_cron_expression"] == "0 0 6 8,9,10 * ?"
    # Same shape as the monthly cron: same seconds, minute and hour; a day-of-month field
    # (a day or a list of days); every month; no day-of-week.
    probe = spec["schedule"]["quartz_cron_expression"].split()
    month = monthly["schedule"]["quartz_cron_expression"].split()
    assert probe[:3] == month[:3] == ["0", "0", "6"]
    assert month[3] == "1" and all(d.isdigit() for d in probe[3].split(","))
    assert probe[4:] == month[4:] == ["*", "?"]
    assert spec["max_concurrent_runs"] == 1
    assert "dependencies" not in spec["environments"][0]["spec"]
    (task,) = spec["tasks"]
    assert task["timeout_seconds"] <= 600 and task["max_retries"] == 0
    params = task["notebook_task"]["base_parameters"]
    assert params["out"] == "/Volumes/workspace/cdm/raw/results/scheduler_probe"
    assert params["trigger_type"] == "{{job.trigger.type}}"
    source = (REPO / "databricks" / "notebooks" / "scheduler_probe.py").read_text("utf-8")
    for forbidden in ("spark", "saveAsTable", "gold_", "import cdm", "from cdm"):
        assert forbidden not in source, forbidden
    # The notebook's default folder is the same, and OUT is the only place it writes.
    assert f'"out": "{params["out"]}"' in source
    assert source.count("write_text(") == 1 and "(OUT / " in source


def test_the_deploy_guard_refuses_a_paused_or_changed_schedule() -> None:
    """Item 1: after reset or create the job is read back; a paused, missing or different
    schedule stops the deploy with a message, a matching one passes."""
    check = deploy_module().check_deployed_schedule
    spec = json.loads((REPO / "databricks" / "monthly_drift_job.json").read_text("utf-8"))
    good = {"settings": {"schedule": dict(spec["schedule"])}}
    check(spec, good)
    paused = {"settings": {"schedule": {**spec["schedule"], "pause_status": "PAUSED"}}}
    with pytest.raises(SystemExit, match="PAUSED, not UNPAUSED"):
        check(spec, paused)
    other = {"settings": {"schedule": {**spec["schedule"], "timezone_id": "America/New_York"}}}
    with pytest.raises(SystemExit, match="timezone_id"):
        check(spec, other)
    with pytest.raises(SystemExit, match="without its schedule"):
        check(spec, {"settings": {}})
    unscheduled = {k: v for k, v in spec.items() if k != "schedule"}
    check(unscheduled, {"settings": {}})
    with pytest.raises(SystemExit, match="does not have"):
        check(unscheduled, good)


def test_upsert_reads_the_job_back_after_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    module = deploy_module()
    spec = json.loads((REPO / "databricks" / "monthly_drift_job.json").read_text("utf-8"))
    calls: list[tuple[str, ...]] = []

    def cli(*args: str, ok_if: str | None = None) -> str:
        calls.append(args)
        if args[:2] == ("jobs", "list"):
            return json.dumps([{"job_id": 7}])
        if args[:2] == ("jobs", "get"):
            return json.dumps(
                {"settings": {"schedule": {**spec["schedule"], "pause_status": "PAUSED"}}}
            )
        return "{}"

    (REPO / "dist").mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="not UNPAUSED"):
        module.upsert_job(cli, spec, dry_run=False)
    assert [c[:2] for c in calls] == [("jobs", "list"), ("jobs", "reset"), ("jobs", "get")]
