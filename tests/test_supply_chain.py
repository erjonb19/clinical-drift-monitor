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


def test_every_action_is_pinned_to_a_commit() -> None:
    """Backlog item 34: every action, GitHub's own included, is pinned to a full commit SHA
    (a tag can be moved to other code), with the release it was taken from in a comment."""
    for path in WORKFLOWS.glob("*.yml"):
        for line in path.read_text("utf-8").splitlines():
            found = re.search(r"uses:\s*([^\s#]+)(.*)", line)
            if found is None or found.group(1).startswith(("./", "docker://")):
                continue  # local actions and container images are not pinned this way
            ref, comment = found.groups()
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref), f"{path.name}: {ref}"
            assert re.match(r"\s*# v\d", comment), f"{path.name}: {ref} names no release"


def test_the_audit_and_scan_run_in_ci() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text("utf-8")
    docker = (WORKFLOWS / "docker.yml").read_text("utf-8")
    assert "pip-audit" in ci and "vulnerable_requirements.txt" in ci
    assert "trivy-action" in docker and "severity: CRITICAL,HIGH" in docker  # item 39
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
    # #83 (2026-11-11): fastapi, open_clip_torch, transformers; #108: mpmath (sympy needs < 1.4)
    for held in ("fastapi", "open_clip_torch", "transformers", "mpmath"):
        assert sorted(_ignored(pip, held)) == [
            "version-update:semver-major",
            "version-update:semver-minor",
        ], held
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


def test_the_base_image_is_pinned_by_digest() -> None:
    """Backlog item 35: a re-pushed tag cannot change the image's base layers unseen."""
    froms = re.findall(r"^FROM\s+(\S+)", (REPO / "Dockerfile").read_text("utf-8"), re.M)
    assert froms and all(re.search(r"@sha256:[0-9a-f]{64}$", f) for f in froms), froms


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


def test_every_workflow_reads_only_unless_it_says_why() -> None:
    """Backlog item 28: each workflow sets a top-level permissions block, the default is
    contents: read, and no checkout leaves the token in .git/config."""
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text("utf-8")
        top = re.search(r"^permissions:\n((?:  .*\n)+)", text, re.MULTILINE)
        assert top is not None, f"{path.name}: no top-level permissions"
        assert top.group(1).strip() == "contents: read", f"{path.name}: {top.group(1)!r}"
        checkouts = text.count("uses: actions/checkout@")
        assert text.count("persist-credentials: false") == checkouts, path.name


def test_every_dependabot_ecosystem_has_a_cooldown() -> None:
    """Backlog item 38: each ecosystem waits at least 7 days after a release before proposing it."""
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    blocks = re.split(r"\n  - package-ecosystem: ", text)[1:]
    assert len(blocks) == 3, "expected the pip, docker and github-actions ecosystems"
    for block in blocks:
        name = block.split("\n", 1)[0]
        days = re.search(r"\n    cooldown:\n      default-days: (\d+)\n", block)
        assert days, f"{name}: no cooldown"
        assert int(days.group(1)) >= 7, f"{name}: cooldown under 7 days"


def test_the_api_image_drops_setuptools_after_installing() -> None:
    """Item 39: the base image's setuptools carries old jaraco.context and wheel copies with
    fixable HIGH CVEs; the API never imports it, so the image removes it after the installs. The
    image also applies Debian's security updates while still root."""
    lines = (REPO / "Dockerfile").read_text("utf-8").splitlines()
    installs = [i for i, line in enumerate(lines) if line.startswith("RUN pip install")]
    drop = lines.index("RUN pip uninstall -y setuptools wheel")
    assert installs and drop > max(installs)
    # Debian's security fixes since the base image was built (apt, zlib1g had fixable HIGHs),
    # from a fixed snapshot.debian.org timestamp, so the build is reproducible (owner, 2026-11-11).
    upgrade = next(i for i, line in enumerate(lines) if "apt-get" in line and "upgrade -y" in line)
    assert upgrade < lines.index("USER demo")
    text = "\n".join(lines)
    for arg in ("DEBIAN_SNAPSHOT", "DEBIAN_SECURITY_SNAPSHOT"):
        assert re.search(rf"^ARG {arg}=\d{{8}}T\d{{6}}Z$", text, re.M), arg
    assert "https://snapshot.debian.org/archive/debian-security/${DEBIAN_SECURITY_SNAPSHOT}" in text
    assert '! grep -rqs "deb.debian.org"' in text  # the build stops if the moving mirror remains


def test_onnxruntime_matches_the_databricks_job_and_its_results() -> None:
    """Owner, 2026-11-11: the serve extra pins the onnxruntime the Databricks ONNX job ran and
    the committed results record; a bump changes all three together, at a Databricks deploy."""
    pyproject = (REPO / "pyproject.toml").read_text("utf-8")
    pin = re.search(r'"onnxruntime==([\d.]+)"', pyproject)
    job = re.search(
        r'"onnxruntime==([\d.]+)"',
        (REPO / "databricks" / "onnx_export_job.json").read_text("utf-8"),
    )
    assert pin and job and pin.group(1) == job.group(1), (pin, job)
    for name in ("results.json", "noise.json"):
        recorded = json.loads((REPO / "results" / "phase3" / "onnx" / name).read_text("utf-8"))
        assert recorded["versions"]["onnxruntime"] == pin.group(1), name
    text = (REPO / ".github" / "dependabot.yml").read_text("utf-8")
    entry = re.search(r"- dependency-name: onnxruntime\n(\s+)(\S+)", text)
    assert entry is not None and not entry.group(2).startswith("update-types")
