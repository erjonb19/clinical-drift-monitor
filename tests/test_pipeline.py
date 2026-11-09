"""Backlog item 17: the pipeline's trigger and the promotion's refusals (cdm.pipeline)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cdm.pipeline import PromotionRefused, trigger_check, verify_promotion

REPO = Path(__file__).resolve().parents[1]
REPLAY = [
    json.loads((REPO / "results/phase3/monthly" / f"month_{m:02d}.json").read_text("utf-8"))
    for m in range(1, 13)
]


def test_the_replay_triggers_only_on_alert_months_whose_labels_arrived() -> None:
    out = trigger_check(REPLAY)
    alerted = {r["month"] for r in REPLAY if r["drift"]["alert"]} | {
        r["month"] for r in REPLAY if any(v["alert"] for v in r["by_skin_type"].values())
    }
    assert out["status"] == "trigger"
    assert set(out["months"]) == alerted - {12}  # month 12's labels arrive after the replay
    assert all(m >= 7 for m in out["months"])  # months 1 to 6 did not alert


def test_no_trigger_says_why() -> None:
    quiet = [r for r in REPLAY if r["month"] <= 6]
    assert trigger_check(quiet) == {"status": "no_trigger", "reason": "no month in alert"}
    last = trigger_check([r for r in REPLAY if r["month"] == 12])
    assert last["status"] == "no_trigger" and "[12]" in last["reason"]


def _decision(promoted: bool) -> bytes:
    return json.dumps({"promoted": promoted, "rule": "spec"}).encode()


def test_a_promotion_needs_the_named_promoting_decision_and_a_checked_bundle() -> None:
    good = _decision(True)
    sha = hashlib.sha256(good).hexdigest()
    tags = {"decision_sha256": sha, "serving_bundle_sha256": "ab" * 32}
    assert verify_promotion(good, sha, tags)["promoted"] is True
    with pytest.raises(PromotionRefused, match="not the one given"):
        verify_promotion(good, "00" * 32, tags)
    bad = _decision(False)
    with pytest.raises(PromotionRefused, match="did not promote"):
        verify_promotion(bad, hashlib.sha256(bad).hexdigest(), tags)
    with pytest.raises(PromotionRefused, match="not registered from"):
        verify_promotion(good, sha, {**tags, "decision_sha256": "cd" * 32})
    with pytest.raises(PromotionRefused, match="serving bundle"):
        verify_promotion(good, sha, {"decision_sha256": sha})


def test_the_promotion_job_runs_only_by_hand_with_the_owners_inputs() -> None:
    """No schedule and no trigger: running it, with the version and the decision's hash, is the
    owner's approval. Its approval inputs are empty in the file, so they must be given."""
    spec = json.loads((REPO / "databricks/promote_challenger_job.json").read_text("utf-8"))
    assert "schedule" not in spec and "trigger" not in spec and spec["max_concurrent_runs"] == 1
    task = spec["tasks"][0]
    assert task["max_retries"] == 0
    params = task["notebook_task"]["base_parameters"]
    for name in ("version", "expect_current", "decision_path", "decision_sha256"):
        assert params[name] == ""
    notebook = (REPO / "databricks/notebooks/promote_challenger.py").read_text("utf-8")
    assert "verify_promotion(" in notebook and "plan_move(" in notebook
