"""Regenerate docs/service_metrics.md: how the research data service is doing.

Four parts, each read from a source, never typed in:

- **Requests:** turnaround of data-request issues, from a snapshot of the GitHub issue
  tracker committed at results/research/g4_metrics/issues.json (refreshed with ``gh`` unless
  ``--offline``). Issues labelled ``synthetic`` are counted separately and marked as such.
- **Cohort service:** runs per phenotype (reuse), refusals by reason, and time from first
  request to approval, from the committed audit log of the cohort service.
- **Output quality:** OMOP gate results of the latest committed build, and the concept map's
  match quality.
- **Model performance:** the rows in config/service_metrics.json, each read from a committed
  results file, with the commit date of that file.

    python scripts/service_metrics.py            (refreshes the issue snapshot; needs gh)
    python scripts/service_metrics.py --offline  (renders from the committed snapshot)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "service_metrics.md"
AUDIT = REPO / "results" / "research" / "g2_cohort" / "audit.jsonl"
BUILD = REPO / "results" / "research" / "a_omop" / "build.json"
CONCEPT_CHECK = REPO / "results" / "research" / "a_omop" / "concept_check.json"
SPEC = REPO / "config" / "service_metrics.json"
ISSUES = REPO / "results" / "research" / "g4_metrics" / "issues.json"
ISSUE_LIMIT = 500


def _time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def request_metrics(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and turnaround (days, created to closed) of data-request issues."""
    out: dict[str, Any] = {}
    for kind in ("real", "synthetic"):
        chosen = [
            i for i in issues
            if ("synthetic" in {lb["name"] for lb in i["labels"]}) == (kind == "synthetic")
        ]  # fmt: skip
        days = [
            (_time(i["closedAt"]) - _time(i["createdAt"])).total_seconds() / 86400
            for i in chosen
            if i.get("closedAt")
        ]
        out[kind] = {
            "opened": len(chosen),
            "closed": len(days),
            "median_days_to_close": round(statistics.median(days), 1) if days else None,
            "max_days_to_close": round(max(days), 1) if days else None,
        }
    return out


def cohort_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Runs per phenotype version, refusals by reason, and hours from the first event on a
    definition to its approval."""
    runs = Counter(f"{e['phenotype']} v{e['version']}" for e in entries if e["event"] == "ran")
    refusals = Counter(e["reason"].split(" (")[0] for e in entries if e["event"] == "refused")
    first: dict[str, datetime] = {}
    waits = []
    for e in entries:
        key = e.get("sha256")
        if not key:
            continue
        first.setdefault(key, _time(e["ts"]))
        if e["event"] == "decision" and e.get("decision") == "approve":
            waits.append((_time(e["ts"]) - first[key]).total_seconds() / 3600)
    return {
        "events": len(entries),
        "runs_by_phenotype": dict(sorted(runs.items())),
        "refusals_by_reason": dict(sorted(refusals.items())),
        "approvals": len(waits),
        "median_hours_to_approval": round(statistics.median(waits), 2) if waits else None,
    }


def lookup(data: Any, path: str) -> float:
    for part in path.split("/"):
        data = data[part]
    return float(data)


def first_commit_date(path: str) -> str:
    """The date ``path`` was first committed. Needs full history: a shallow clone gives the
    date of its oldest commit instead."""
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%as", "--", path],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    return out[-1] if out else "uncommitted"


def model_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in spec["model_performance"]:
        data = json.loads((REPO / row["file"]).read_text(encoding="utf-8"))
        rows.append(
            {
                **{k: row[k] for k in ("model", "evaluated_on", "metric", "file")},
                "value": lookup(data, row["value"]),
                "low": lookup(data, row["low"]),
                "high": lookup(data, row["high"]),
                "committed": first_commit_date(row["file"]),
            }
        )
    return rows


def render(
    requests: dict[str, Any] | None,
    cohort: dict[str, Any] | None,
    build: dict[str, Any] | None,
    concepts: dict[str, Any],
    models: list[dict[str, Any]],
    snapshot_utc: str | None = None,
) -> str:
    lines = [
        "# Service metrics",
        "",
        "Generated by `scripts/service_metrics.py`. Do not edit by hand. Every number comes from "
        "the issue tracker, the cohort service's audit log, or a committed results file. This "
        "is a portfolio service on public research data; synthetic requests are counted "
        "separately and labelled.",
        "",
        "## Requests (turnaround)",
        "",
    ]
    if requests is None:
        lines.append("No snapshot of the issue tracker committed yet.")
    else:
        lines += [f"From the issue tracker snapshot taken {snapshot_utc} "
                  "(`results/research/g4_metrics/issues.json`).", ""]  # fmt: skip
        lines += ["| Requests | Opened | Closed | Median days to close | Longest |",
                  "| --- | ---: | ---: | ---: | ---: |"]  # fmt: skip
        for kind, m in requests.items():
            synthetic = kind == "synthetic"
            label = "Synthetic (labelled `synthetic`)" if synthetic else "Not labelled synthetic"
            med = "n/a" if m["median_days_to_close"] is None else m["median_days_to_close"]
            top = "n/a" if m["max_days_to_close"] is None else m["max_days_to_close"]
            lines.append(f"| {label} | {m['opened']} | {m['closed']} | {med} | {top} |")
    lines += ["", "## Cohort service (reuse and refusals)", ""]
    if not cohort or not cohort["events"]:
        lines.append("No audit log committed yet (`results/research/g2_cohort/audit.jsonl`).")
    else:
        lines += [f"{cohort['events']} audited events; {cohort['approvals']} approvals; median "
                  f"time from first request to approval: {cohort['median_hours_to_approval']} "
                  "hours.", "", "| Phenotype | Runs |", "| --- | ---: |"]  # fmt: skip
        lines += [f"| {k} | {v} |" for k, v in cohort["runs_by_phenotype"].items()]
        lines += ["", "| Refusal reason | Count |", "| --- | ---: |"]
        lines += [f"| {k} | {v} |" for k, v in cohort["refusals_by_reason"].items()]
    lines += ["", "## Output quality", ""]
    if build is None:
        lines.append("OMOP build: not yet recorded (`results/research/a_omop/build.json`).")
    else:
        failed = [g["name"] for g in build["gates"] if int(g["violations"])]
        lines.append(
            f"OMOP build `{build['etl_reference']}`: {len(build['gates']) - len(failed)} of "
            f"{len(build['gates'])} gates passed"
            + (f"; failed: {', '.join(failed)}" if failed else "")
            + "."  # fmt: skip
        )
    matches = concepts["rows_by_match"]
    total = sum(matches.values())
    lines += [
        "",
        f"Concept map: {total} source values; " + ", ".join(
            f"{k} {v}" for k, v in sorted(matches.items())) + " (match types, "
        "`config/omop_concepts.csv`).",
        "",
        "## Model performance",
        "",
        "Rows are different test sets, so they are not a trend line: they show where the "
        "shipped model holds and where it does not. Percent, with 95% bootstrap intervals.",
        "",
        "| Model | Evaluated on | Metric | Value | 95% CI | Results file (first committed) |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]  # fmt: skip
    for r in models:
        lines.append(
            f"| {r['model']} | {r['evaluated_on']} | {r['metric']} | {100 * r['value']:.1f} | "
            f"[{100 * r['low']:.1f}, {100 * r['high']:.1f}] | `{r['file']}` ({r['committed']}) |"
        )
    return "\n".join(lines) + "\n"


def issues_from_github() -> list[dict[str, Any]]:
    """Data-request issues: number, open and close times, label names. No titles or text."""
    out = subprocess.run(
        ["gh", "issue", "list", "--label", "data-request", "--state", "all", "--limit",
         str(ISSUE_LIMIT), "--json", "number,createdAt,closedAt,labels"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    issues: list[dict[str, Any]] = json.loads(out)
    if len(issues) >= ISSUE_LIMIT:
        raise SystemExit(f"{len(issues)} issues: the query limit may have cut some off")
    return [{"number": i["number"], "createdAt": i["createdAt"], "closedAt": i["closedAt"],
             "labels": [{"name": lb["name"]} for lb in i["labels"]]} for i in issues]  # fmt: skip


def snapshot(issues: list[dict[str, Any]]) -> dict[str, Any]:
    query = "gh issue list --label data-request --state all"
    return {"queried_utc": datetime.now(UTC).isoformat(timespec="seconds"), "query": query,
            "issues": issues}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        ISSUES.parent.mkdir(parents=True, exist_ok=True)
        ISSUES.write_bytes(
            (json.dumps(snapshot(issues_from_github()), indent=2) + chr(10)).encode()
        )
    taken = json.loads(ISSUES.read_text(encoding="utf-8")) if ISSUES.exists() else None
    requests = request_metrics(taken["issues"]) if taken else None
    entries = (
        [json.loads(x) for x in AUDIT.read_text(encoding="utf-8").splitlines() if x.strip()]
        if AUDIT.exists()
        else []
    )
    build = json.loads(BUILD.read_text(encoding="utf-8")) if BUILD.exists() else None
    concepts = json.loads(CONCEPT_CHECK.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    page = render(requests, cohort_metrics(entries), build, concepts, model_rows(spec),
                  taken["queried_utc"] if taken else None)  # fmt: skip
    PAGE.write_bytes(page.encode())
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
