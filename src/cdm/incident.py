"""Backlog item 8 / Phase 4: drift incident reports in 8D structure, and their review queue.

Two writers produce the same output shape, which :mod:`cdm.report_eval` scores:

- :func:`template_report`: deterministic. Its findings are :func:`cdm.scenarios.expected_findings`
  and its 8D prose quotes only the record's own numbers (rounded to fewer digits, so the
  groundedness check matches them). It needs no model and no key. Because its findings come
  from the same rules the eval's answers do, it is a **baseline**, never evidence that an LLM
  can do the task.
- :func:`gemini_call`: one call to the Gemini API's ``generateContent`` (free tier, owner's
  decision 2026-11-03) with the prompt from :func:`build_prompt`, over the standard library's
  HTTP client (no new dependency). The key comes from ``GEMINI_API_KEY`` and is sent only in a
  header, never logged. A provider failure after retries is an ``outage``; an answer that is not
  a JSON object is ``invalid`` (``cdm.report_eval``'s outcomes).

The review queue is one JSON file per incident under ``results/phase4/review_queue/``. A
report counts as sent only after a person records ``approved`` there (PLAN.md Phase 4: "Reports
land in a human review queue before they count as sent"); :func:`review` appends the decision
and never rewrites the report.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdm.groundedness import check
from cdm.report_eval import EIGHT_D, FINDINGS, correct, shape_errors
from cdm.scenarios import RULES, expected_findings

REPO = Path(__file__).resolve().parents[2]
MODEL = "gemini-2.5-flash"  # fixed before any live call (eval spec); the owner may change it
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRIES, TIMEOUT_S = 3, 60
DECISIONS = {"approved", "rejected"}

SCHEMA = {
    "severity": '"none", "warning" or "alert"',
    "overall_alert": "true or false",
    "alerting_skin_types": "list of skin-type keys whose test alerted",
    "new_sources": 'list of by_source keys other than "client_test" with images this month',
    "containment": "the rules' containment word for the severity",
    "eight_d_report": "true only when the month is in alert",
    "delayed_accuracy_available": "true when delayed_accuracy_of_month is present",
    "accuracy_drop": "null when no delayed accuracy; else true when it is below accuracy_floor",
    "caveats": '"skin_type_recorded_only_at_new_sites", "new_images_counted_not_scored", '
    '"labels_not_yet_arrived", each when it applies',
    "claims": "list of {value, source}: every number the report uses, with the record field "
    "or rule it comes from",
    "eight_d": "null unless eight_d_report; then an object with " + ", ".join(EIGHT_D),
}


def month_context(slo: Path = REPO / "config" / "slo.json") -> dict[str, Any]:
    """The rules context for a real or replayed month: the fixed rules plus the abstain-share
    target from ``config/slo.json``, so a report may say the target is broken. (The staged
    scenarios use the rules alone, as the eval spec fixes.)"""
    target = json.loads(slo.read_text(encoding="utf-8"))["targets"]["abstain_share"]["value"]
    return {**RULES, "abstain_share_target": float(target)}


def build_prompt(record: Mapping[str, Any], rules: Mapping[str, Any], floor: float) -> str:
    """The instruction: the record, the fixed rules context, and the output shape. Nothing
    else; the scenario's ID, family, seed and answers never reach the model."""
    return (
        "You write drift incident reports for a skin-lesion classifier's monthly monitoring "
        "job. Use only the record and the rules below. Every number you write must be copied "
        "from the record or the rules (you may round it); never compute a new number such as a "
        "difference or a ratio. Answer with one JSON object and nothing else, with exactly "
        "these keys:\n"
        + json.dumps(SCHEMA, indent=1)
        + "\n\nRules:\n"
        + json.dumps({"rules": rules, "accuracy_floor": floor}, indent=1)
        + "\n\nRecord (one month of the drift table):\n"
        + json.dumps(record, indent=1)
    )


def _fmt(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}"


def template_report(
    record: Mapping[str, Any], floor: float, abstain_target: float | None = None
) -> dict[str, Any]:
    """A grounded report from the rules alone (see the module docstring: a baseline). With
    ``abstain_target`` (a real or replayed month), it says whether the abstain share broke it."""
    found = expected_findings(dict(record), floor)
    drift = record["drift"]
    month = record["month"]
    claims: list[dict[str, Any]] = [
        {"value": month, "source": "month"},
        {"value": drift["images"], "source": "drift.images"},
        {"value": drift["psi"], "source": "drift.psi"},
        {"value": drift["ks_p"], "source": "drift.ks_p"},
        {"value": record["abstain_share"], "source": "abstain_share"},
    ]
    out: dict[str, Any] = {**found, "claims": claims, "eight_d": None}
    if not found["eight_d_report"]:
        return out
    typed = found["alerting_skin_types"]
    sources = ", ".join(f"{s} ({record['by_source'][s]} images)" for s in found["new_sources"])
    shifted = (
        f"PSI {_fmt(drift['psi'], 2)} with KS p = {drift['ks_p']:.2e}"
        if drift["alert"]
        else "no overall alert"
    )
    typed_text = (
        "; skin type "
        + ", ".join(f"{t} ({record['by_skin_type'][t]['images']} images)" for t in typed)
        + " also alerted, which detects the new site, not anything about skin type"
        if typed
        else ""
    )
    abstain_text = ""
    if abstain_target is not None:
        broken = record["abstain_share"] > abstain_target
        abstain_text = (
            f", {'above' if broken else 'within'} the abstain-share target of "
            f"{_fmt(100 * abstain_target, 0)}%" + (" (target broken)" if broken else "")
        )
    delayed = record.get("delayed_accuracy_of_month")
    accuracy = (
        f"Month {month}'s own accuracy arrives with the next month's run. The latest known, "
        f"month {delayed['month']}'s delayed balanced accuracy, was "
        f"{_fmt(delayed['balanced_accuracy'])} on {delayed['labelled']} labelled images, "
        + ("below" if found["accuracy_drop"] else "not below")
        + f" the in-distribution floor of {_fmt(floor)}."
        if delayed
        else "Labels have not arrived, so accuracy is not known yet; it arrives with a later run."
    )
    out["eight_d"] = {
        "problem": (
            f"Month {month}: the OOD scores of {drift['images']} images shifted ({shifted}"
            f"{typed_text}). Images came from {sources or 'the clients only'}. "
            f"{_fmt(100 * record['abstain_share'], 1)}% of images were above the abstain threshold"
            f"{abstain_text}."
        ),
        "containment": (
            f"Containment word: {found['containment']}. Abstain: the service flags every image "
            "above the abstain threshold for a person to review instead of reading it from the "
            "model. Hold: the champion is not replaced while the alert is open (a runbook "
            "procedure, not enforced by code). The steps are in the runbook's containment section "
            "(docs/runbook.md)."
        ),
        "root_cause": (
            f"Images from {sources or 'a new source'} entered the month's batch. The test shows "
            "the scores moved, not why; the source mix is the leading explanation and must be "
            "confirmed by a reviewer."
        ),
        "permanent_fix": (
            "Relabel a sample from the new source, then train a challenger and promote it only "
            "if it wins under the committed retraining rule."
        ),
        "verification": accuracy,
        "prevention": (
            "Keep the monthly drift test and the per-skin-type test running, and review each "
            "alert's report in the queue before it counts as sent."
        ),
    }
    return out


def gemini_call(
    prompt: str,
    api_key: str,
    model: str = MODEL,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """One generation. Returns ``outcome`` (ok / outage / invalid), the parsed ``output`` (ok
    only), the provider ``error`` (outage only), tokens and latency. Never raises on a provider
    failure, and never puts the key in what it returns. A 400, 401, 403 or 404 (a wrong key,
    model or request) raises: it is a setup error, and counting it as an outage would hide it."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }).encode()  # fmt: skip
    request = urllib.request.Request(  # noqa: S310 (a fixed https endpoint)
        ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    start, error, payload, attempt_ms = time.perf_counter(), "", None, 0
    for attempt in range(RETRIES):
        tried = time.perf_counter()
        try:
            with opener(request, timeout=TIMEOUT_S) as response:
                payload = json.loads(response.read())
            attempt_ms = round(1000 * (time.perf_counter() - tried))
            break
        except urllib.error.HTTPError as err:
            if err.code in (400, 401, 403, 404):
                # The request or the key is wrong: a setup error, not an outage. Stop the run.
                raise RuntimeError(f"Gemini refused the request (HTTP {err.code})") from None
            error = f"HTTP {err.code}"
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as err:
            # URLError, timeouts, connection resets and broken responses: the provider failed.
            error = type(err).__name__
        if attempt < RETRIES - 1:
            sleep(2.0 * 2**attempt)
    # latency_ms: the answering attempt alone; wall_ms: every attempt and back-off sleep.
    wall_ms = round(1000 * (time.perf_counter() - start))
    if payload is None:
        return {"outcome": "outage", "output": None, "error": error, "wall_ms": wall_ms}
    latency_ms = attempt_ms
    usage = payload.get("usageMetadata", {})
    tokens = {
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "thinking_tokens": usage.get("thoughtsTokenCount"),
        "wall_ms": wall_ms,
    }
    try:
        text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
        output = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {"outcome": "invalid", "output": None, "latency_ms": latency_ms, **tokens}
    outcome = "invalid" if shape_errors(output) else "ok"
    return {"outcome": outcome, "output": output, "latency_ms": latency_ms, **tokens}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def incident(
    record: Mapping[str, Any], report: Mapping[str, Any], floor: float, writer: str, kind: str
) -> dict[str, Any]:
    """A queue entry: the record it was written from, the report, its groundedness and its
    agreement with the rules' findings, and an empty review history. ``kind`` says whether the
    drift is real, replayed or staged (CLAUDE.md: never mixed)."""
    if kind not in ("real", "replayed", "staged"):
        raise ValueError("kind is real, replayed or staged")
    grounded = check(report, record, month_context(), floor)
    expected = expected_findings(dict(record), floor)
    return {
        "month": record["month"],
        "kind": kind,
        "writer": writer,
        "record": record,
        "report": report,
        "accuracy_floor": floor,
        "groundedness": grounded,
        "findings_match_rules": correct(report, expected),
        "expected_findings": {k: expected[k] for k in (*FINDINGS, "caveats")},
        "status": "pending_review",
        "reviews": [],
        "queued_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def review(entry: dict[str, Any], decision: str, reviewer: str, note: str) -> dict[str, Any]:
    """Append a person's decision. An ungrounded report can be rejected but never approved."""
    if decision not in DECISIONS:
        raise ValueError(f"decision is one of {sorted(DECISIONS)}")
    if not reviewer.strip():
        raise ValueError("a review names its reviewer")
    # Re-run the check rather than trust the stored flag: the file may have been edited.
    recheck = check(entry["report"], entry["record"], month_context(), entry["accuracy_floor"])
    if decision == "approved" and not (recheck["grounded"] and entry["groundedness"]["grounded"]):
        raise ValueError("an ungrounded report cannot be approved")
    entry["reviews"].append({
        "decision": decision, "reviewer": reviewer.strip(), "note": note,
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
    })  # fmt: skip
    entry["status"] = decision
    return entry


def save(entry: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
