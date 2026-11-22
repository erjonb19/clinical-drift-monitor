"""Backlog item 23: the project's Claude Code settings keep the standard's deny list."""

from __future__ import annotations

import fnmatch
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


def _denied(command: str) -> bool:
    """Whether a Bash deny rule matches, with `*` as a wildcard over the whole command text
    (Claude Code's Bash rule matching; deny wins over allow). Claude Code also splits compound
    commands and denies if any part matches; this emulation does not split, so it is stricter."""
    rules = [r[5:-1] for r in SETTINGS["permissions"]["deny"] if r.startswith("Bash(")]
    return any(fnmatch.fnmatchcase(command, rule) for rule in rules)


def test_shell_reads_of_env_files_and_find_deletes_are_denied() -> None:
    """Owner, 2026-11-11: the read-only allow list (cat, grep, head, tail) must not reach .env
    files, and find must not delete or run commands. Rules match the command's text, so a read
    that never names the file (grep -r KEY .) is outside what they can catch."""
    for command in (
        "cat .env",
        "cat ./.env",
        "cat app/.env.local",
        "head -n 5 .env",
        "tail .env.production",
        "grep KEY .env",
        "grep -n KEY .env other.txt",
        "find . -name '*.pyc' -delete",
        "find . -exec rm {} ;",
        "find . -type f -execdir cat {} ;",
        "cat '.env'",
        'grep KEY ".env"',
        "find -delete",
        "find -exec rm {} +",
        "find . -okdir rm {} ;",
        "find . -fprint out.txt",
    ):
        assert _denied(command), command
    for command in ("cat README.md", "grep -n env src/cdm/extract.py"):
        assert not _denied(command), command
    # .env.example is caught too, like the standard's Read(./.env.*): fail closed.
    assert _denied("cat .env.example")


def test_curl_and_gh_api_writes_are_denied_whatever_the_flag_position() -> None:
    """Item 23's check (2026-11-19): `Bash(curl * -X *)` needs text before the flag, so
    `curl -X POST <url>` ran in a cloud session (it failed only to connect). Each curl and
    `gh api` write rule now has a flag-first twin and a no-space twin (`-XPOST`,
    `--request=POST`), approved by the owner on 2026-11-20 (owner step S13)."""
    for command in (
        "curl -X POST https://example.com",
        "curl -XPOST https://example.com",
        "curl -sS https://example.com -XPOST",
        "curl --request=POST https://example.com",
        "curl --request PUT https://example.com",
        "curl -d a=1 https://example.com",
        "curl -da=1 https://example.com",
        "curl --data-binary @f https://example.com",
        "curl -F x=@f https://example.com",
        "curl --form=x=1 https://example.com",
        "curl -T f https://example.com",
        "curl --upload-file=f https://example.com",
        "curl --json {} https://example.com",
        "gh api -X POST repos/o/r/issues",
        "gh api --method=DELETE repos/o/r",
        "gh api -fa=b repos/o/r/issues",
        "gh api -F x=@f repos/o/r",
        "gh api --input body.json repos/o/r",
        "gh api --raw-field=x repos/o/r",
    ):
        assert _denied(command), command
    for command in (
        "curl -sS https://example.com",
        "curl -sSfL -o out.json https://example.com",
        "gh api repos/o/r/pulls",
        "gh api --paginate repos/o/r/pulls --jq .[].number",
    ):
        assert not _denied(command), command
