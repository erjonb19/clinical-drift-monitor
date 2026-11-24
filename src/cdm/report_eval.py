"""Phase 4 (backlog item 8): score incident-report runs against the approved eval spec.

``results/phase4/eval_spec.json`` (thresholds approved by the owner, 2026-11-24), applied to
recorded runs. Nothing here calls a model; a generator writes the runs, and CI replays them.

- **A run** is ``{"scenario": id, "run": n, "outcome": "ok" | "outage" | "invalid", "output":
  ...}``. An ``outage`` (the provider failed after retries) is not scored. An ``invalid`` run (the
  provider answered, but not in the required shape) is scored and is never correct.
- **Correct:** every structured finding equals the known one (lists compared as sets), every
  expected caveat is stated, and the 8D report is present, with all six sections non-empty,
  exactly when one is due.
- **Grounded:** :func:`cdm.groundedness.check` over the whole output. One ungrounded scored run
  anywhere fails the eval (a hard gate).
- **A scenario passes** when at least 2 of its 3 runs are correct; one left with fewer than 2
  scored runs counts as not passed.
- **The verdict:** ``fail`` on any ungrounded run; else ``inconclusive`` (exit 2, never a pass)
  when more than 3 scenarios have fewer than 2 scored runs; else ``pass`` when at least 27 of 30
  scenarios pass; else ``fail``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from cdm.groundedness import check

FINDINGS = (
    "severity",
    "overall_alert",
    "alerting_skin_types",
    "new_sources",
    "containment",
    "eight_d_report",
    "delayed_accuracy_available",
    "accuracy_drop",
)
LIST_FIELDS = {"alerting_skin_types", "new_sources"}
EIGHT_D = ("problem", "containment", "root_cause", "permanent_fix", "verification", "prevention")
OUTCOMES = {"ok", "outage", "invalid"}
RUNS, MIN_CORRECT = 3, 2
PASS_AT_LEAST, MAX_SHORT = 27, 3


def shape_errors(output: Any) -> list[str]:
    """Why an output is not in the required shape (empty when it is)."""
    if not isinstance(output, Mapping):
        return ["the output is not a JSON object"]
    errors = [f"missing {k}" for k in (*FINDINGS, "caveats", "claims") if k not in output]
    for k in LIST_FIELDS | {"caveats", "claims"}:
        if k in output and not isinstance(output[k], list):
            errors.append(f"{k} is not a list")
    eight_d = output.get("eight_d")
    if eight_d is not None and not isinstance(eight_d, Mapping):
        errors.append("eight_d is not an object")
    return errors


def correct(output: Any, expected: Mapping[str, Any]) -> bool:
    """The run's structured findings, caveats and 8D report against the known findings."""
    if shape_errors(output):
        return False
    for k in FINDINGS:
        got, want = output[k], expected[k]
        if k in LIST_FIELDS:
            if not all(isinstance(x, str) for x in got) or set(got) != set(want):
                return False
        elif type(got) is not type(want) or got != want:
            return False
    if not set(expected["caveats"]) <= {c for c in output["caveats"] if isinstance(c, str)}:
        return False
    eight_d = output.get("eight_d")
    written = isinstance(eight_d, Mapping) and all(
        isinstance(eight_d.get(s), str) and eight_d[s].strip() for s in EIGHT_D
    )
    if expected["eight_d_report"]:
        return written
    return eight_d is None


def score(
    runs: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
    floor: float,
) -> dict[str, Any]:
    """The eval's result for recorded runs. Refuses runs that name an unknown scenario, an
    unknown outcome, a repeated or out-of-range run number."""
    by_id = {s["id"]: s for s in scenarios}
    per: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for r in runs:
        key = (r["scenario"], r["run"])
        if r["scenario"] not in by_id or r["outcome"] not in OUTCOMES:
            raise ValueError(f"unknown scenario or outcome in run {key}")
        if key in seen or not (isinstance(r["run"], int) and 1 <= r["run"] <= RUNS):
            raise ValueError(f"run {key} is repeated or out of range")
        seen.add(key)
        per[r["scenario"]].append(r)

    rows, ungrounded = [], []
    for sid, s in by_id.items():
        scored = [r for r in per.get(sid, []) if r["outcome"] != "outage"]
        n_correct = 0
        for r in scored:
            g = check(r.get("output"), s["record"], rules, floor)
            if not g["grounded"]:
                ungrounded.append({"scenario": sid, "run": r["run"], "unmatched": g["unmatched"]})
            if r["outcome"] == "ok" and correct(r.get("output"), s["expected"]):
                n_correct += 1
        short = len(scored) < MIN_CORRECT
        rows.append({
            "scenario": sid, "scored_runs": len(scored), "correct_runs": n_correct,
            "short_of_runs": short, "passed": not short and n_correct >= MIN_CORRECT,
        })  # fmt: skip
    passed = sum(r["passed"] for r in rows)
    short = sum(r["short_of_runs"] for r in rows)
    if ungrounded:
        verdict = "fail"
    elif short > MAX_SHORT:
        verdict = "inconclusive"
    else:
        verdict = "pass" if passed >= PASS_AT_LEAST else "fail"
    return {
        "verdict": verdict,
        "exit_code": {"pass": 0, "fail": 1, "inconclusive": 2}[verdict],
        "scenarios_passed": passed,
        "scenarios": len(rows),
        "scenarios_short_of_runs": short,
        "ungrounded_runs": ungrounded,
        "outages": sum(r["outcome"] == "outage" for r in runs),
        "per_scenario": rows,
    }
