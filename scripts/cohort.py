"""The governed cohort service from the command line (research Phase G2).

    python scripts/cohort.py check                       every phenotype: valid?, hash, status
    python scripts/cohort.py decide FILE approve --reviewer NAME --reason TEXT [--request URL]
    python scripts/cohort.py run FILE --requester NAME [--request URL] [--engine databricks]
    python scripts/cohort.py verify                      the audit log's hash chain

Approvals go to phenotypes/approvals.jsonl (committed: who approved which exact definition).
Pass --approvals to keep a separate file, as the synthetic request exercise does. Decisions in
such a file authorise runs only for requesters marked "synthetic: ...".
The audit log defaults to logs/cohort_audit.jsonl (not committed); pass --audit to keep one
with the results. The databricks engine queries workspace.cdm.omop_* through the SQL
warehouse; the duckdb engine reads OMOP tables saved as parquet in --data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from cdm.cohort import (
    OMOP_TABLES,
    Approvals,
    AuditLog,
    CohortService,
    Engine,
    PhenotypeError,
    Refused,
    duckdb_engine,
    load,
    omop_table_map,
    verify_chain,
)

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "phenotypes"


def databricks_engine() -> Engine:
    spec = importlib.util.spec_from_file_location(
        "evidence_export", REPO / "scripts" / "phase1_export_evidence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sql: Any = module.Sql()

    def run(statement: str) -> pd.DataFrame:
        frame: pd.DataFrame = sql(statement)
        # The statement API returns strings; everything but the criterion text is a number.
        for column in frame.columns:
            if column != "criterion":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    return run


def service(args: argparse.Namespace) -> CohortService:
    if getattr(args, "engine", "duckdb") == "databricks":
        engine, tables = databricks_engine(), omop_table_map("workspace.cdm.omop_")
        source = "databricks workspace.cdm.omop_*"
    else:
        frames = {t: pd.read_parquet(Path(args.data) / f"{t}.parquet") for t in OMOP_TABLES}
        engine, tables = duckdb_engine(frames)
        source = f"duckdb {args.data}"
    return CohortService(engine, tables, Approvals(Path(args.approvals)),
                         AuditLog(Path(args.audit)), source)  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--audit", default=str(REPO / "logs" / "cohort_audit.jsonl"))
    parser.add_argument("--approvals", default=str(LIBRARY / "approvals.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    decide = sub.add_parser("decide")
    decide.add_argument("file", type=Path)
    decide.add_argument("decision", choices=["approve", "reject", "escalate"])
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--reason", required=True)
    decide.add_argument("--request", default="")
    run = sub.add_parser("run")
    run.add_argument("file", type=Path)
    run.add_argument("--requester", required=True)
    run.add_argument("--request", default="")
    run.add_argument("--engine", choices=["duckdb", "databricks"], default="duckdb")
    run.add_argument("--data", default=str(REPO / "data" / "omop"))
    sub.add_parser("verify")
    args = parser.parse_args()

    approvals = Approvals(Path(args.approvals))
    if args.command == "check":
        bad = 0
        for path in sorted(LIBRARY.glob("*/v*.yaml")):
            try:
                p = load(path)
                print(f"{p.name} v{p.version}  {p.sha256[:12]}  {approvals.status(p)}")
            except PhenotypeError as exc:
                bad += 1
                print(f"INVALID {path}: {exc}")
        return 1 if bad else 0
    if args.command == "verify":
        problems = verify_chain(AuditLog(Path(args.audit)).entries())
        print("\n".join(problems) or "audit chain intact")
        return 1 if problems else 0
    if args.command == "decide":
        audit = AuditLog(Path(args.audit))
        svc = CohortService(lambda s: pd.DataFrame(), {}, approvals, audit)
        print(json.dumps(asdict(svc.decide(args.file, args.decision, args.reviewer, args.reason,
                                           args.request)), indent=1))  # fmt: skip
        return 0
    stand_in = Path(args.approvals).resolve() != (LIBRARY / "approvals.jsonl").resolve()
    if stand_in and not args.requester.startswith("synthetic:"):
        print("REFUSED: a separate approvals file only serves synthetic requests", file=sys.stderr)
        return 2
    try:
        result = service(args).run(args.file, args.requester, args.request)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
