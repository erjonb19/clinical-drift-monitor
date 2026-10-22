"""The request issue form is valid: GitHub drops a broken form silently, and the request
service depends on it."""

from __future__ import annotations

from pathlib import Path

import yaml

FORMS = Path(__file__).resolve().parents[1] / ".github" / "ISSUE_TEMPLATE"


def test_request_form_has_unique_ids_labels_and_required_fields() -> None:
    form = yaml.safe_load((FORMS / "data_request.yml").read_text(encoding="utf-8"))
    assert {"name", "description", "body"} <= set(form)
    assert all(isinstance(label, str) for label in form["labels"])
    assert "data-request" in form["labels"]
    fields = [item for item in form["body"] if item["type"] != "markdown"]
    ids = [item["id"] for item in fields]
    assert len(ids) == len(set(ids))
    for item in fields:
        assert item["type"] in {"input", "textarea", "dropdown", "checkboxes"}
        assert item["attributes"]["label"]
    required = {i["id"] for i in fields if i.get("validations", {}).get("required")}
    assert {"requester", "question", "cohort", "approval", "output"} <= required


def test_template_chooser_config_is_valid() -> None:
    config = yaml.safe_load((FORMS / "config.yml").read_text(encoding="utf-8"))
    assert isinstance(config.get("blank_issues_enabled"), bool)
    for link in config.get("contact_links", []):
        assert {"name", "url", "about"} <= set(link)
