"""Backlog item 23: the project's Claude Code settings keep the standard's deny list."""

from __future__ import annotations

import json
from pathlib import Path

SETTINGS = json.loads(
    (Path(__file__).resolve().parents[1] / ".claude" / "settings.json").read_text("utf-8")
)
# docs/SECURITY_STANDARDS.md §17.2's deny list, kept whole.
STANDARD_DENY = {
    "Read(./.env)", "Read(./.env.*)", "Read(./**/.env)", "Read(./**/.env.*)",
    "Read(./secrets/**)", "Read(./**/*.pem)", "Read(./**/*.key)", "Read(./**/*.tfstate)",
    "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(~/.config/gcloud/**)", "Read(~/.kube/**)",
    "Read(~/.npmrc)", "Read(~/.pypirc)", "Bash(git push --force *)", "Bash(git push -f *)",
    "Bash(rm -rf *)", "Bash(sudo *)",
}  # fmt: skip


def test_the_standards_deny_list_is_kept() -> None:
    assert set(SETTINGS["permissions"]["deny"]) >= STANDARD_DENY


def test_bypass_mode_is_disabled_and_settings_are_protected() -> None:
    perms = SETTINGS["permissions"]
    assert perms["disableBypassPermissionsMode"] == "disable"
    assert "Edit(./.claude/**)" in perms["deny"]
    assert not set(perms["allow"]) & set(perms["deny"])
