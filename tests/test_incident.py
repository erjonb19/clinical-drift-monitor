"""Backlog item 8: incident reports (template and Gemini writers) and the review queue."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from cdm.incident import build_prompt, gemini_call, incident, review, template_report
from cdm.report_eval import score
from cdm.scenarios import RULES, expected_findings, in_distribution_accuracy_floor

REPO = Path(__file__).resolve().parents[1]
MONTHS = [
    json.loads((REPO / f"results/phase3/monthly/month_{n:02d}.json").read_text(encoding="utf-8"))
    for n in range(1, 13)
]
FLOOR = in_distribution_accuracy_floor()
DATA = json.loads((REPO / "results/phase4/scenarios.json").read_text(encoding="utf-8"))


def test_the_template_report_is_grounded_and_right_for_every_committed_month() -> None:
    for record in MONTHS:
        entry = incident(record, template_report(record, FLOOR), FLOOR, "template", "replayed")
        assert entry["groundedness"]["grounded"], (record["month"], entry["groundedness"])
        assert entry["findings_match_rules"], record["month"]
        assert (entry["report"]["eight_d"] is not None) == entry["expected_findings"][
            "eight_d_report"
        ]


def test_the_template_writer_passes_the_eval_but_is_only_a_baseline() -> None:
    """Its findings are the rules' own, so it scores 30 of 30: proof the scorer and the writer
    agree, not evidence about any LLM."""
    runs = [
        {"scenario": s["id"], "run": n, "outcome": "ok",
         "output": template_report(s["record"], DATA["accuracy_floor"])}
        for s in DATA["scenarios"] for n in (1, 2, 3)
    ]  # fmt: skip
    result = score(runs, DATA["scenarios"], DATA["rules"], DATA["accuracy_floor"])
    assert (result["verdict"], result["scenarios_passed"]) == ("pass", 30)


def test_the_prompt_holds_the_record_and_rules_but_no_answer() -> None:
    s = DATA["scenarios"][0]
    prompt = build_prompt(s["record"], DATA["rules"], DATA["accuracy_floor"])
    assert json.dumps(s["record"], indent=1) in prompt and "alert_rule" in prompt
    for secret in (s["id"], s["family"], str(s["seed"]), "expected"):
        assert secret not in prompt, secret


class Fake:
    """Answers urlopen calls from a list: a dict (JSON body), bytes, or an exception."""

    def __init__(self, *answers: Any) -> None:
        self.answers: list[Any] = list(answers)
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        body = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
        return io.BytesIO(body)


def http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "x", {}, None)  # type: ignore[arg-type]


def gemini(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 300}}  # fmt: skip


def test_a_call_is_ok_invalid_or_an_outage_and_never_returns_the_key() -> None:
    record = MONTHS[8]
    good = json.dumps(template_report(record, FLOOR))
    fake = Fake(gemini(good))
    call = gemini_call("p", "SECRET-KEY", opener=fake, sleep=lambda s: None)
    assert call["outcome"] == "ok" and call["input_tokens"] == 900
    assert fake.requests[0].get_header("X-goog-api-key") == "SECRET-KEY"
    assert "SECRET-KEY" not in json.dumps(call)
    assert (
        gemini_call("p", "k", opener=Fake(gemini("not json")), sleep=lambda s: None)["outcome"]
        == "invalid"
    )
    assert (
        gemini_call("p", "k", opener=Fake(gemini("{}")), sleep=lambda s: None)["outcome"]
        == "invalid"
    )
    outage = gemini_call(
        "p", "k", opener=Fake(http(503), http(429), TimeoutError()), sleep=lambda s: None
    )
    assert (outage["outcome"], outage["error"], outage["output"]) == (
        "outage",
        "TimeoutError",
        None,
    )
    recovered = gemini_call("p", "k", opener=Fake(http(503), gemini(good)), sleep=lambda s: None)
    assert recovered["outcome"] == "ok"


def test_a_setup_error_stops_the_run_instead_of_counting_as_an_outage() -> None:
    for code in (400, 401, 403, 404):
        with pytest.raises(RuntimeError, match=f"HTTP {code}"):
            gemini_call("p", "k", opener=Fake(http(code)), sleep=lambda s: None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gemini_call("p", "")


def test_review_needs_a_reviewer_and_never_approves_an_ungrounded_report() -> None:
    record = MONTHS[8]
    bad = template_report(record, FLOOR) | {"claims": [{"value": 0.987654, "source": "drift.psi"}]}
    entry = incident(record, bad, FLOOR, "template", "replayed")
    with pytest.raises(ValueError, match="ungrounded"):
        review(entry, "approved", "owner", "")
    with pytest.raises(ValueError, match="reviewer"):
        review(entry, "rejected", " ", "")
    assert review(entry, "rejected", "owner", "a made-up PSI")["status"] == "rejected"
    with pytest.raises(ValueError, match="real, replayed or staged"):
        incident(record, bad, FLOOR, "template", "live")


def test_the_queued_month_9_report_is_grounded_replayed_and_awaits_review() -> None:
    entry = json.loads((REPO / "results/phase4/review_queue/month_09_template.json").read_text())
    assert (entry["kind"], entry["writer"], entry["status"]) == (
        "replayed",
        "template",
        "pending_review",
    )
    assert entry["record"] == MONTHS[8] and entry["groundedness"]["grounded"]
    assert entry["report"] == template_report(MONTHS[8], FLOOR)
    assert entry["expected_findings"]["severity"] == "alert"
    assert entry["report"]["severity"] == expected_findings(MONTHS[8], FLOOR)["severity"]
    words: dict[str, str] = RULES["containment_words"]  # type: ignore[assignment]
    assert words["alert"] == entry["report"]["containment"]


def test_the_script_scores_recorded_runs_and_never_rewrites_a_queued_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "incident_report", REPO / "scripts/incident_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    runs = tmp_path / "runs.jsonl"
    rows = [
        {"scenario": s["id"], "run": n, "outcome": "ok",
         "output": template_report(s["record"], DATA["accuracy_floor"])}
        for s in DATA["scenarios"] for n in (1, 2, 3)
    ]  # fmt: skip
    runs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert module.main(["score", str(runs)]) == 0
    with pytest.raises(SystemExit, match="never rewritten"):
        module.main(["write", "--month", "9", "--kind", "replayed"])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        module.main(["eval", "--out", str(tmp_path / "e")])
