"""Backlog item 8: write, review, evaluate and score drift incident reports.

  python scripts/incident_report.py write --month 9 --kind replayed           # template writer
  python scripts/incident_report.py write --month 9 --kind replayed --writer gemini   # needs key
  python scripts/incident_report.py review <queue file> --decision approved --reviewer NAME
  python scripts/incident_report.py eval --out results/phase4/eval          # 90 Gemini calls
  python scripts/incident_report.py score results/phase4/eval/runs.jsonl    # replay, no calls

``write`` reads a committed month record (``results/phase3/monthly/month_<n>.json``) and adds a
report to the review queue (``results/phase4/review_queue/``). ``eval`` makes the eval's 3 runs
of each of the 30 staged scenarios with Gemini, logs every call (model, prompt hash, tokens,
latency, outcome, dollar cost 0 on the free tier) and scores them; ``score`` scores recorded runs
without calling anything and exits 0 (pass), 1 (fail) or 2 (inconclusive). Live calls read
``GEMINI_API_KEY`` from the environment; the key is never printed or written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cdm.incident import (  # noqa: E402
    MODEL,
    build_prompt,
    gemini_call,
    incident,
    month_context,
    prompt_sha256,
    review,
    save,
    template_report,
)
from cdm.report_eval import RUNS, score  # noqa: E402
from cdm.scenarios import in_distribution_accuracy_floor  # noqa: E402

MONTHLY = REPO / "results" / "phase3" / "monthly"
QUEUE = REPO / "results" / "phase4" / "review_queue"
SCENARIOS = REPO / "results" / "phase4" / "scenarios.json"


def _key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    return key


def cmd_write(args: argparse.Namespace) -> int:
    path = QUEUE / f"month_{args.month:02d}_{args.writer}.json"
    if path.exists():
        sys.exit(f"{path.relative_to(REPO)} exists: a queued report is never rewritten")
    record = json.loads((MONTHLY / f"month_{args.month:02d}.json").read_text(encoding="utf-8"))
    floor, context = in_distribution_accuracy_floor(), month_context()
    if args.writer == "template":
        report = template_report(record, floor, context["abstain_share_target"])
        meta: dict[str, Any] = {}
    else:
        prompt = build_prompt(record, context, floor)
        call = gemini_call(prompt, _key(), args.model)
        if call["outcome"] != "ok":
            sys.exit(f"no report: the call was {call['outcome']} ({call.get('error', '')})")
        report = call["output"]
        meta = {"model": args.model, "prompt_sha256": prompt_sha256(prompt)}
        meta |= {k: call.get(k) for k in ("latency_ms", "wall_ms", "input_tokens", "output_tokens",
                                          "thinking_tokens")}  # fmt: skip
    entry = incident(record, report, floor, args.writer, args.kind) | {"call": meta}
    save(entry, path)
    print(json.dumps({"queued": str(path.relative_to(REPO)), **entry["groundedness"],
                      "findings_match_rules": entry["findings_match_rules"]}))  # fmt: skip
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    path = Path(args.file)
    entry = json.loads(path.read_text(encoding="utf-8"))
    save(review(entry, args.decision, args.reviewer, args.note), path)
    print(f"{path.name}: {entry['status']}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cmd_eval(args: argparse.Namespace) -> int:
    """Resumable: runs already recorded as ok or invalid are kept, outages are tried again, and
    at most ``--max-calls`` calls are made, so a free tier's daily quota can spread the 90 calls
    over several days. Each run is written as soon as it finishes. Scores once all 90 exist."""
    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    key, out = _key(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path, calls_path = out / "runs.jsonl", out / "calls.jsonl"
    kept = {
        (r["scenario"], r["run"]): r for r in _read_jsonl(runs_path) if r["outcome"] != "outage"
    }
    outages = {
        (r["scenario"], r["run"]): r for r in _read_jsonl(runs_path) if r["outcome"] == "outage"
    }
    made = 0
    for s in data["scenarios"]:
        prompt = build_prompt(s["record"], data["rules"], data["accuracy_floor"])
        for n in range(1, RUNS + 1):
            if (s["id"], n) in kept or made >= args.max_calls:
                continue
            time.sleep(args.pause)  # stay under the free tier's requests-per-minute limit
            call = gemini_call(prompt, key, args.model)
            made += 1
            run = {
                "scenario": s["id"],
                "run": n,
                "outcome": call["outcome"],
                "output": call["output"],
            }
            if call["outcome"] == "outage":
                run["error"] = call["error"]
                outages[(s["id"], n)] = run
            else:
                kept[(s["id"], n)] = run
                outages.pop((s["id"], n), None)
            rows = sorted(
                [*kept.values(), *outages.values()], key=lambda r: (r["scenario"], r["run"])
            )
            runs_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            log = {"model": args.model, "prompt_sha256": prompt_sha256(prompt), "scenario": s["id"],
                   "run": n, "outcome": call["outcome"], "error": call.get("error"),
                   **{k: call.get(k) for k in ("input_tokens", "output_tokens", "thinking_tokens",
                                               "latency_ms", "wall_ms")},
                   "dollar_cost": 0.0}  # fmt: skip
            with calls_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")
    recorded = len(kept) + len(outages)
    if recorded < RUNS * len(data["scenarios"]):
        print(f"partial: {recorded} of {RUNS * len(data['scenarios'])} runs recorded; run again")
        return 3
    return _score(runs_path, out / "result.json")


def _score(runs_path: Path, out: Path | None) -> int:
    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line]
    result = score(runs, data["scenarios"], data["rules"], data["accuracy_floor"])
    summary = {k: v for k, v in result.items() if k != "per_scenario"}
    if out is not None:
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return int(result["exit_code"])


def cmd_score(args: argparse.Namespace) -> int:
    return _score(Path(args.runs), Path(args.out) if args.out else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--month", type=int, required=True)
    w.add_argument("--kind", choices=("real", "replayed", "staged"), required=True)
    w.add_argument("--writer", choices=("template", "gemini"), default="template")
    w.add_argument("--model", default=MODEL)
    r = sub.add_parser("review")
    r.add_argument("file")
    r.add_argument("--decision", choices=("approved", "rejected"), required=True)
    r.add_argument("--reviewer", required=True)
    r.add_argument("--note", default="")
    e = sub.add_parser("eval")
    e.add_argument("--out", default=str(REPO / "results" / "phase4" / "eval"))
    e.add_argument("--model", default=MODEL)
    e.add_argument("--pause", type=float, default=7.0, help="seconds between calls")
    e.add_argument("--max-calls", type=int, default=90, help="calls this time (a daily quota)")
    sc = sub.add_parser("score")
    sc.add_argument("runs")
    sc.add_argument("--out", default="")
    args = parser.parse_args(argv)
    handler = {"write": cmd_write, "review": cmd_review, "eval": cmd_eval, "score": cmd_score}
    return handler[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
