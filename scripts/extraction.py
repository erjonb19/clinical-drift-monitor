"""Clinical text extraction (research Phase G5) from the command line.

    python scripts/extraction.py sheet [--csv PATH] [--n 100]   labelling sheet for you to fill
    python scripts/extraction.py run --split dev|scoring --method llm|keyword [--limit N]
    python scripts/extraction.py score                          scoring split, both methods
    python scripts/extraction.py smoke                          one Haiku call on a made-up note
    python scripts/extraction.py spend                          LLM spend so far

Everything with note text goes to the work folder, outside the repository: the labelling
sheet, the LLM cache, the predictions, the spend log and the error sheet. The default is
~/data/mtsamples/work (set CDM_NOTES_DIR to change it). Only ``score`` writes into the repo:
results/research/g5_extraction/score.json, aggregates only.

Protocol (docs/EVAL_PLAN.md): the dev split is for prompt design; the first LLM run on the
scoring split freezes the prompt hash, and a later prompt is refused there. A score from fewer
than 50 fully labelled scoring notes, or from predictions made with another prompt, is
written but marked not reportable. LLM spend is logged in ~/.cdm/llm_spend.jsonl for every
work folder, so the $5 cap cannot be reset by switching folders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cdm.extract import (
    FIELDS,
    MODEL,
    MTSamples,
    Note,
    SpendLog,
    anthropic_transport,
    choices,
    error_kind,
    keyword_extract,
    llm_extract,
    outside_repo,
    prompt_hash,
    score,
    write_sheet,
)
from cdm.omop import load_concept_map

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("CDM_NOTES_DIR", Path.home() / "data" / "mtsamples" / "work"))
OUT = REPO / "results" / "research" / "g5_extraction" / "score.json"
# One spend log for every work folder, so the cap cannot be reset by switching folders.
SPEND = Path.home() / ".cdm" / "llm_spend.jsonl"
MIN_SCORING_NOTES = 50
SMOKE_NOTE = Note(
    "smoke", "made-up", "Dermatology", "smoke test",
    "Made-up note for a connection test. 61-year-old woman, 6 mm pigmented papule on the left "
    "forearm. Excisional biopsy: compound nevus, completely excised.",
)  # fmt: skip


def sheet_rows() -> pd.DataFrame:
    return pd.read_csv(WORK / "labelling_sheet.csv", dtype=str, keep_default_na=False)


def as_note(row: pd.Series) -> Note:
    return Note(row["note_id"], row["source"], row["specialty"], row["title"], row["text"])


def predictions_path(method: str, split: str) -> Path:
    return WORK / f"predictions_{method}_{split}.csv"


def run(split: str, method: str, limit: int | None) -> None:
    allowed = choices(load_concept_map(REPO / "config" / "omop_concepts.csv"))
    rows = sheet_rows()
    rows = rows[rows["split"] == split]
    if limit is not None:
        rows = rows.head(limit)
    spend = SpendLog(SPEND)
    if method == "llm" and split == "scoring":
        frozen = WORK / "scoring_prompt_sha.txt"
        sha = prompt_hash(allowed)
        if frozen.exists() and frozen.read_text().strip() != sha:
            first = frozen.read_text().strip()
            sys.exit(f"the scoring split was first run with prompt {first}; this prompt is "
                     f"{sha}. A changed prompt cannot be scored (protocol).")  # fmt: skip
        frozen.write_text(sha)
    send = anthropic_transport() if method == "llm" else None
    sha = prompt_hash(allowed) if method == "llm" else "keyword"
    out = []
    for _, row in rows.iterrows():
        note = as_note(row)
        if send is None:
            got = keyword_extract(note, allowed)
        else:
            got = llm_extract(note, allowed, send, spend, WORK / "llm_cache.jsonl")
        out.append({"note_id": note.note_id, **got, "prompt_sha": sha})
    pd.DataFrame(out, columns=["note_id", *FIELDS, "prompt_sha"]).to_csv(
        predictions_path(method, split), index=False
    )
    print(f"{len(out)} notes; LLM spend so far ${spend.total():.4f}")


def score_all() -> None:
    allowed = choices(load_concept_map(REPO / "config" / "omop_concepts.csv"))
    rows = sheet_rows()
    scoring = rows[rows["split"] == "scoring"]
    filled = (scoring[[f"label_{f}" for f in FIELDS]] != "").all(axis=1)
    labelled = scoring[filled]
    bad = [
        f"{r['note_id']} {f}: {r[f'label_{f}']!r}"
        for _, r in labelled.iterrows()
        for f in FIELDS
        if r[f"label_{f}"] not in allowed.labels(f)
    ]
    if bad:
        sys.exit("labels not in labelling_choices.csv (fix them first): " + "; ".join(bad))
    reasons = []
    if len(labelled) < MIN_SCORING_NOTES:
        reasons.append(
            f"{len(labelled)} fully labelled scoring notes, fewer than {MIN_SCORING_NOTES}"
        )
    result: dict[str, Any] = {
        "scored_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "notes_source": sorted(set(labelled["source"])),
        "labelled_scoring_notes": len(labelled),
        "reportable": None,
        "not_reportable_because": reasons,
        "model": MODEL,
        "prompt_sha": None,
        "methods": {},
    }
    errors = []
    for method in ("llm", "keyword"):
        path = predictions_path(method, "scoring")
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("note_id")
        preds = {nid: dict(frame.loc[nid]) for nid in labelled["note_id"] if nid in frame.index}
        missing = len(labelled) - len(preds)
        if missing:
            sys.exit(f"{method}: {missing} labelled scoring notes have no prediction; run first")
        if method == "llm":
            shas = {p["prompt_sha"] for p in preds.values()}
            frozen = WORK / "scoring_prompt_sha.txt"
            first = frozen.read_text().strip() if frozen.exists() else None
            result["prompt_sha"] = sorted(shas)
            if len(shas) != 1 or first is None or shas != {first}:
                reasons.append(f"LLM predictions come from prompts {sorted(shas)}, but the "
                               f"scoring split was first run with {first}")  # fmt: skip
        result["methods"][method] = score(labelled, preds, allowed)
        for _, row in labelled.iterrows():
            for field in FIELDS:
                truth, guess = row[f"label_{field}"], preds[row["note_id"]][field]
                if error_kind(field, truth, guess, allowed) is not None:
                    errors.append({"method": method, "note_id": row["note_id"], "field": field,
                                   "label": truth, "predicted": guess, "text": row["text"],
                                   "label_error (you)": ""})  # fmt: skip
    result["reportable"] = not reasons
    spend = SpendLog(SPEND)
    result["llm_calls"] = len(spend.entries())
    result["llm_spend_usd"] = round(spend.total(), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    pd.DataFrame(errors).to_csv(outside_repo(WORK) / "errors.csv", index=False)
    print(f"wrote {OUT} (reportable: {result['reportable']}) and {WORK / 'errors.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("sheet")
    sheet.add_argument("--csv", type=Path, default=Path.home() / "data/mtsamples/mtsamples.csv")
    sheet.add_argument("--n", type=int, default=100)
    go = sub.add_parser("run")
    go.add_argument("--split", choices=["dev", "scoring"], required=True)
    go.add_argument("--method", choices=["llm", "keyword"], required=True)
    go.add_argument("--limit", type=int)
    sub.add_parser("score")
    sub.add_parser("smoke")
    sub.add_parser("spend")
    args = parser.parse_args()
    outside_repo(WORK)
    if args.command == "sheet":
        allowed = choices(load_concept_map(REPO / "config" / "omop_concepts.csv"))
        path = write_sheet(MTSamples(args.csv).notes(), WORK, allowed, args.n)
        print(f"wrote {path} and {WORK / 'labelling_choices.csv'}")
    elif args.command == "run":
        run(args.split, args.method, args.limit)
    elif args.command == "score":
        score_all()
    elif args.command == "smoke":
        allowed = choices(load_concept_map(REPO / "config" / "omop_concepts.csv"))
        spend = SpendLog(SPEND)
        got = llm_extract(SMOKE_NOTE, allowed, anthropic_transport(), spend,
                          WORK / "llm_cache.jsonl")  # fmt: skip
        print(json.dumps(got), f"spend so far ${spend.total():.6f}")
    else:
        spend = SpendLog(SPEND)
        print(f"{len(spend.entries())} calls, ${spend.total():.6f}")


if __name__ == "__main__":
    main()
