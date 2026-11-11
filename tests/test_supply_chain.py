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
            if ref.startswith(("./", "docker://")):
                continue  # local actions and container images are not pinned this way
            owner, _, version = ref.partition("@")
            owner = owner.split("/")[0]
            if owner not in {"actions", "github"}:
                assert re.fullmatch(r"[0-9a-f]{40}", version), f"{path.name}: {ref}"


def test_the_audit_and_scan_run_in_ci() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text("utf-8")
    docker = (WORKFLOWS / "docker.yml").read_text("utf-8")
    assert "pip-audit" in ci and "vulnerable_requirements.txt" in ci
    assert "trivy-action" in docker and "severity: CRITICAL" in docker
    assert (REPO / ".github" / "dependabot.yml").exists() and (WORKFLOWS / "codeql.yml").exists()


def test_every_trivyignore_entry_has_a_reason_above_it() -> None:
    lines = (REPO / ".trivyignore").read_text("utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith("#"):
            assert i > 0 and lines[i - 1].startswith("#"), f"no reason above {line}"


def _ignored(text: str, name: str) -> list[str]:
    """The update types ignored for one dependency in dependabot.yml (plain text, no YAML lib)."""
    match = re.search(rf"- dependency-name: {re.escape(name)}\n\s+update-types: \[([^\]]+)\]", text)
    return [t.strip() for t in match.group(1).split(",")] if match else []


def test_dependabot_keeps_python_311_and_the_model_stack_on_patches() -> None:
    """Owner, 2026-11-09: no Python minor or major bump of the image (PR #66 tried 3.15), and
    patch-only updates for the stack behind committed results (PR #68 tried flwr 1.43.0)."""
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    both = ["version-update:semver-major", "version-update:semver-minor"]
    docker = text.split("package-ecosystem: docker", 1)[1].split("package-ecosystem:", 1)[0]
    assert sorted(_ignored(docker, "python")) == both
    pip = text.split("package-ecosystem: pip", 1)[1].split("package-ecosystem:", 1)[0]
    stack = ["numpy", "scikit-learn", "flwr", "mlflow", "mlflow-skinny"]
    for name in stack:
        assert sorted(_ignored(pip, name)) == both, name


def test_the_image_keeps_the_cpu_torch_wheels() -> None:
    """A Dependabot pip PR (#68) dropped the `+cpu` label from torch and torchvision in the
    image's constraints, which would pull the CUDA wheels; and it cannot see the Dockerfile's
    RUN line, so the two pins could disagree."""
    pins = dict(
        line.split("==", 1)
        for line in (REPO / "docker" / "constraints.txt").read_text("utf-8").splitlines()
        if line.startswith(("torch==", "torchvision=="))
    )
    assert set(pins) == {"torch", "torchvision"}
    dockerfile = (REPO / "Dockerfile").read_text("utf-8")
    for name, version in pins.items():
        assert version.endswith("+cpu"), f"{name}=={version}"
        assert f"{name}=={version.removesuffix('+cpu')}" in dockerfile, f"{name} differs"


def test_dependabot_keeps_floors_and_the_uvicorn_that_flwr_needs() -> None:
    """Owner, 2026-11-10, after #74: floors stay unless a release needs them raised, and
    uvicorn gets patches only while flwr is held at 1.39.x (it needs uvicorn < 0.50)."""
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    pip = text.split("package-ecosystem: pip", 1)[1].split("package-ecosystem:", 1)[0]
    assert "versioning-strategy: increase-if-necessary" in pip
    assert sorted(_ignored(pip, "uvicorn")) == [
        "version-update:semver-major",
        "version-update:semver-minor",
    ]


def test_dependabot_never_updates_torch() -> None:
    """Owner, 2026-11-10: torch and torchvision are ignored outright (no update types), since
    Dependabot's bumps drop +cpu and miss the Dockerfile; a bump is made by hand."""
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    pip = text.split("package-ecosystem: pip", 1)[1].split("package-ecosystem:", 1)[0]
    for name in ("torch", "torchvision"):
        entry = re.search(rf"- dependency-name: {name}\n(\s+)(\S+)", pip)
        assert entry is not None, name
        assert not entry.group(2).startswith("update-types"), name


def test_secret_scanning_runs_pinned_with_a_self_test() -> None:
    """Backlog item 25: gitleaks in CI is a checksummed pinned release that must catch a
    planted key, and the pre-commit hook pins the same version by commit."""
    text = (WORKFLOWS / "secrets.yml").read_text("utf-8")
    version = re.search(r'GITLEAKS_VERSION: "([\d.]+)"', text)
    assert version and re.search(r'GITLEAKS_SHA256: "[0-9a-f]{64}"', text)
    assert "sha256sum -c" in text and "planted key" in text and "fetch-depth: 0" in text
    # Steps run under bash -e: without set +e the self-test stops at gitleaks' exit 1 and
    # always fails (review of #88).
    assert "set +e" in text.split("planted key", 1)[1].split("- name:", 1)[0]
    hook = (REPO / ".pre-commit-config.yaml").read_text("utf-8")
    assert re.search(rf"rev: [0-9a-f]{{40}} # v{re.escape(version.group(1))}", hook)
    config = (REPO / ".gitleaks.toml").read_text("utf-8")
    assert "useDefault = true" in config and 'condition = "AND"' in config
