"""Backlog item 11: supply-chain settings stay in force."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"


def test_every_accepted_vulnerability_has_a_reason_and_a_review_date() -> None:
    accepted = json.loads((REPO / "config" / "pip_audit_ignore.json").read_text("utf-8"))[
        "accepted"
    ]
    for entry in accepted:
        assert entry["ids"] and len(entry["reason"]) > 40
        date.fromisoformat(entry["review_by"])


def test_third_party_actions_are_pinned_to_a_commit() -> None:
    """GitHub's own actions (actions/*, github/*) may use tags; anyone else's must be a SHA."""
    for path in WORKFLOWS.glob("*.yml"):
        for ref in re.findall(r"uses:\s*([^\s#]+)", path.read_text("utf-8")):
            owner, version = ref.split("/")[0], ref.split("@")[1]
            if owner not in {"actions", "github"}:
                assert re.fullmatch(r"[0-9a-f]{40}", version), f"{path.name}: {ref}"


def test_the_audit_and_scan_run_in_ci() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text("utf-8")
    docker = (WORKFLOWS / "docker.yml").read_text("utf-8")
    assert "pip-audit" in ci and "vulnerable_requirements.txt" in ci
    assert "trivy-action" in docker and "severity: CRITICAL" in docker
    assert (REPO / ".github" / "dependabot.yml").exists() and (WORKFLOWS / "codeql.yml").exists()
