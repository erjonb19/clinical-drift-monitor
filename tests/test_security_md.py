"""Backlog item 42: SECURITY.md points to private reporting and carries no email address."""

from __future__ import annotations

import re
from pathlib import Path

TEXT = (Path(__file__).resolve().parents[1] / "SECURITY.md").read_text(encoding="utf-8")


def test_reports_go_through_private_vulnerability_reporting() -> None:
    assert "security/advisories/new" in TEXT and "Report a vulnerability" in TEXT


def test_no_email_address_is_published() -> None:
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", TEXT)
