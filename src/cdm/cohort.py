"""Governed cohort and feasibility counts over OMOP CDM tables (research Phase G2).

A requester never writes SQL. A cohort is a versioned phenotype file (``phenotypes/``) in a
small allowlisted language: an index condition, extra criteria on conditions, procedures,
images and sex, and an optional breakdown. The service:

1. **validates** the file against the allowlist: only known keys, only integer concept IDs,
   only allowlisted breakdowns. Anything else is refused;
2. runs it only if a recorded human **approved that exact definition** (by SHA-256 of its
   canonical content). A changed definition needs a new approval;
3. **compiles** it to SQL against generic OMOP CDM v5.4 table and column names, so it runs on
   DuckDB (tests, local), on Databricks SQL (``workspace.cdm.omop_*``), or on any other OMOP
   that supplies a table map;
4. **suppresses small cells**: a count from 1 to 10 is hidden, and so are other cells where
   subtraction would recover it. Every hidden cell carries the same label, with no stated
   range, because a range ("under 11", "11 or more") is itself enough to pin a count;
5. writes every request, refusal, approval and run to a **hash-chained audit log** that
   :func:`verify_chain` checks.

The pattern (allowlist, human approval, audit) follows the author's
Security-Constrained-Agent-Runtime; this is fresh code.

Semantics: criteria are person-level, as in OMOP cohort tools. A person with a melanoma and
a histopathology procedure on another lesion meets "melanoma" and "histopathology". Age is
at the index condition: year of the condition start date minus year of birth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

MIN_CELL = 11
SUPPRESSED = "suppressed"
# A cell hidden only so a small one cannot be recovered. Same label on purpose: two labels
# would tell a reader which hidden cells are under 11.
COMPLEMENT = SUPPRESSED
OMOP_TABLES = (
    "person",
    "condition_occurrence",
    "procedure_occurrence",
    "image_occurrence",
)
BREAKDOWNS = {
    "gender_concept_id": "p.gender_concept_id",
    "age_band": "CAST(FLOOR(f.index_age / 10) * 10 AS INTEGER)",
}
TOP_KEYS = {"name", "version", "title", "purpose", "author", "index", "include", "exclude",
            "breakdown"}  # fmt: skip
INDEX_KEYS = {"condition_concept_ids", "age_min", "age_max"}
CRITERION_KEYS = {
    "condition_concept_ids": ("condition_occurrence", "condition_concept_id"),
    "procedure_concept_ids": ("procedure_occurrence", "procedure_concept_id"),
    "anatomic_site_concept_ids": ("image_occurrence", "anatomic_site_concept_id"),
    "modality_concept_ids": ("image_occurrence", "modality_concept_id"),
    "gender_concept_ids": ("person", "gender_concept_id"),
}
APPROVE, REJECT, ESCALATE = "approve", "reject", "escalate"
DECISIONS = {APPROVE, REJECT, ESCALATE}  # escalate hands the item on; it is not a verdict

Engine = Callable[[str], pd.DataFrame]


class PhenotypeError(ValueError):
    """The phenotype file is outside the allowlisted language."""


@dataclass(frozen=True)
class Phenotype:
    name: str
    version: int
    title: str
    definition: dict[str, Any]
    sha256: str
    path: str


def canonical(definition: Mapping[str, Any]) -> str:
    return json.dumps(definition, sort_keys=True, separators=(",", ":"))


def _concept_list(value: object, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PhenotypeError(f"{where}: must be a non-empty list of concept IDs")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value):
        raise PhenotypeError(f"{where}: concept IDs must be positive integers, got {value!r}")
    return sorted(set(value))


def _criterion(raw: object, where: str) -> dict[str, list[int]]:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise PhenotypeError(f"{where}: each criterion has exactly one key")
    ((key, value),) = raw.items()
    if key not in CRITERION_KEYS:
        raise PhenotypeError(f"{where}: {key!r} is not an allowed criterion")
    return {key: _concept_list(value, f"{where}.{key}")}


def parse(definition: Mapping[str, Any], path: str = "<memory>") -> Phenotype:
    """Validate a phenotype against the allowlist and return it with its hash."""
    unknown = set(definition) - TOP_KEYS
    if unknown:
        raise PhenotypeError(f"unknown keys: {sorted(unknown)}")
    for key in ("name", "version", "title", "index"):
        if key not in definition:
            raise PhenotypeError(f"missing {key!r}")
    version = definition["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PhenotypeError("version must be a positive integer")
    index = definition["index"]
    if not isinstance(index, dict) or set(index) - INDEX_KEYS:
        raise PhenotypeError(f"index: allowed keys are {sorted(INDEX_KEYS)}")
    clean_index: dict[str, Any] = {
        "condition_concept_ids": _concept_list(index.get("condition_concept_ids"), "index")
    }
    for bound in ("age_min", "age_max"):
        if bound in index:
            if not isinstance(index[bound], int) or not 0 <= index[bound] <= 120:
                raise PhenotypeError(f"index.{bound}: an integer age from 0 to 120")
            clean_index[bound] = index[bound]
    if clean_index.get("age_min", 0) > clean_index.get("age_max", 120):
        raise PhenotypeError("index: age_min is above age_max")
    clean: dict[str, Any] = {
        "name": str(definition["name"]),
        "version": definition["version"],
        "title": str(definition["title"]),
        "purpose": str(definition.get("purpose", "")),
        "author": str(definition.get("author", "")),
        "index": clean_index,
    }
    for part in ("include", "exclude"):
        raw = definition.get(part, []) or []
        if not isinstance(raw, list):
            raise PhenotypeError(f"{part}: must be a list")
        clean[part] = [_criterion(c, f"{part}[{i}]") for i, c in enumerate(raw)]
    breakdown = definition.get("breakdown", []) or []
    if not isinstance(breakdown, list) or set(breakdown) - set(BREAKDOWNS):
        raise PhenotypeError(f"breakdown: allowed values are {sorted(BREAKDOWNS)}")
    clean["breakdown"] = list(breakdown)
    digest = hashlib.sha256(canonical(clean).encode()).hexdigest()
    return Phenotype(clean["name"], clean["version"], clean["title"], clean, digest, path)


def load(path: Path) -> Phenotype:
    """A phenotype file: ``phenotypes/<name>/v<version>.yaml``, and the name and version in
    the file must match its path, so a file cannot pose as another version."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PhenotypeError(f"{path}: not a mapping")
    pheno = parse(raw, str(path.as_posix()))
    if path.parent.name != pheno.name or path.stem != f"v{pheno.version}":
        raise PhenotypeError(f"{path}: expected {pheno.name}/v{pheno.version}.yaml")
    return pheno


# ---- SQL ------------------------------------------------------------------------------------


def omop_table_map(prefix: str) -> dict[str, str]:
    """Table names for an OMOP where each table is ``<prefix><table>``, for example
    ``workspace.cdm.omop_`` (this repo) or ``omop.`` (a schema)."""
    return {t: f"{prefix}{t}" for t in OMOP_TABLES}


def _ids(values: list[int]) -> str:
    # Only integers validated by parse() reach SQL text, so this cannot inject.
    return ", ".join(str(int(v)) for v in values)


def _criterion_sql(criterion: dict[str, list[int]], tables: Mapping[str, str]) -> str:
    ((key, ids),) = criterion.items()
    table, column = CRITERION_KEYS[key]
    return f"SELECT DISTINCT person_id FROM {tables[table]} WHERE {column} IN ({_ids(ids)})"


def compile_sql(pheno: Phenotype, tables: Mapping[str, str]) -> dict[str, str]:
    """SQL for the attrition table (persons left after each step) and, if asked for, the
    breakdown of the final cohort. Plain SQL that DuckDB and Spark SQL both run."""
    d = pheno.definition
    idx = d["index"]
    age = "YEAR(c.condition_start_date) - p.year_of_birth"
    where = [f"c.condition_concept_id IN ({_ids(idx['condition_concept_ids'])})"]
    if "age_min" in idx:
        where.append(f"{age} >= {int(idx['age_min'])}")
    if "age_max" in idx:
        where.append(f"{age} <= {int(idx['age_max'])}")
    ctes = [
        "step0 AS (SELECT c.person_id, MIN(" + age + ") AS index_age "
        f"FROM {tables['condition_occurrence']} c JOIN {tables['person']} p "
        f"ON c.person_id = p.person_id WHERE {' AND '.join(where)} GROUP BY c.person_id)"
    ]
    labels = ["index condition" + (" with age limits" if len(where) > 1 else "")]
    steps = [("include", c) for c in d["include"]] + [("exclude", c) for c in d["exclude"]]
    for i, (kind, criterion) in enumerate(steps, start=1):
        op = "IN" if kind == "include" else "NOT IN"
        ctes.append(
            f"step{i} AS (SELECT person_id, index_age FROM step{i - 1} "
            f"WHERE person_id {op} ({_criterion_sql(criterion, tables)}))"
        )
        ((key, ids),) = criterion.items()
        labels.append(f"{kind} {key} {ids}")
    last = f"step{len(steps)}"
    with_clause = "WITH " + ",\n".join(ctes)
    attrition = (
        with_clause
        + "\n"
        + "\nUNION ALL\n".join(
            f"SELECT {i} AS step, '{label}' AS criterion, COUNT(*) AS persons FROM step{i}"
            for i, label in enumerate(labels)
        )
        + "\nORDER BY step"
    )
    out = {"attrition": attrition}
    if d["breakdown"]:
        dims = [f"{BREAKDOWNS[b]} AS {b}" for b in d["breakdown"]]
        out["breakdown"] = (
            f"{with_clause}\nSELECT {', '.join(dims)}, COUNT(*) AS persons FROM {last} f "
            f"JOIN {tables['person']} p ON f.person_id = p.person_id "
            f"GROUP BY {', '.join(str(i + 1) for i in range(len(dims)))} "
            f"ORDER BY {', '.join(str(i + 1) for i in range(len(dims)))}"
        )
    return out


# ---- small cells ----------------------------------------------------------------------------


def suppress(counts: pd.Series, total: int | None = None) -> list[str]:
    """Counts as strings with cells from 1 to 10 hidden. If the total is published and any
    cell is small, every non-zero cell is hidden: partial hiding (the smallest other cells)
    was tried first, and a brute-force test showed that a reader who knows the rule can undo
    it. Zero is shown: it reveals no person. Ask for a coarser breakdown to see more."""
    values = [int(v) for v in counts]
    small = any(0 < v < MIN_CELL for v in values)
    hide_all = total is not None and small
    return [SUPPRESSED if (0 < v < MIN_CELL) or (hide_all and v > 0) else str(v) for v in values]


def suppress_breakdown(counts: pd.Series, final: int, final_shown: str) -> tuple[list[str], str]:
    """A breakdown of the final cohort, and how the final count may be shown with it. The
    cells add up to the final count, so when it is hidden every non-zero cell is hidden too.
    When the cells are hidden and the count is shown, ``k`` hidden cells adding up to it
    could each still be 11 or more only if it is at least ``k + 10``; below that the final
    count is hidden as well."""
    values = [int(v) for v in counts]
    if not final_shown.isdigit():
        return [str(v) if v == 0 else SUPPRESSED for v in values], final_shown
    cells = suppress(counts, total=final)
    hidden = sum(1 for c in cells if c == SUPPRESSED)
    if hidden and final < hidden + MIN_CELL - 1:
        return [str(v) if v == 0 else SUPPRESSED for v in values], SUPPRESSED
    return cells, final_shown


def suppress_attrition(persons: pd.Series) -> list[str]:
    """Attrition counts: a step from 1 to 10 is hidden. The drop between two shown steps is
    also a count of people, so a step is hidden too ("suppressed") while its drop from the last
    *shown* step is from 1 to 10; hiding one step does not make the next one safe."""
    values = [int(v) for v in persons]
    out = []
    shown: int | None = None
    for v in values:
        drop = shown - v if shown is not None else 0
        if 0 < v < MIN_CELL:
            out.append(SUPPRESSED)
        elif 0 < drop < MIN_CELL:
            out.append(COMPLEMENT)
        else:
            out.append(str(v))
            shown = v
    return out


# ---- approvals and audit --------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Decision:
    name: str
    version: int
    sha256: str
    decision: str
    reviewer: str
    reason: str
    request: str = ""
    decided_utc: str = field(default_factory=_now)


class Approvals:
    """Decisions on phenotype versions, one JSON line each (committed: who approved what)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def all(self) -> list[Decision]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Decision(**json.loads(line)) for line in lines if line.strip()]

    def record(self, decision: Decision) -> None:
        if decision.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as f:
            f.write((json.dumps(asdict(decision)) + "\n").encode())

    def status(self, pheno: Phenotype) -> str:
        """The latest verdict on this exact definition: approve, reject, or pending (no
        verdict, or only escalations)."""
        verdicts = [
            d.decision
            for d in self.all()
            if d.name == pheno.name and d.sha256 == pheno.sha256 and d.decision != ESCALATE
        ]
        return verdicts[-1] if verdicts else "pending"


class AuditLog:
    """Append-only JSON lines; each entry carries the hash of the one before, so an edited
    or deleted line breaks :func:`verify_chain`. Dropping the last lines does not: nothing
    outside the log anchors its head (a limit, listed in phenotypes/README.md)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        prior = self.entries()
        entry = {"seq": len(prior), "ts": _now(), "event": event, **fields,
                 "prev": prior[-1]["hash"] if prior else ""}  # fmt: skip
        entry["hash"] = hashlib.sha256(canonical(entry).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as f:
            f.write((json.dumps(entry) + "\n").encode())
        return entry


def verify_chain(entries: list[dict[str, Any]]) -> list[str]:
    """Problems in an audit chain: a broken link, an edited entry, or a gap in ``seq``."""
    problems, prev = [], ""
    for i, entry in enumerate(entries):
        body = {k: v for k, v in entry.items() if k != "hash"}
        if entry.get("seq") != i:
            problems.append(f"entry {i}: seq {entry.get('seq')}")
        if body.get("prev") != prev:
            problems.append(f"entry {i}: does not follow the entry before")
        if hashlib.sha256(canonical(body).encode()).hexdigest() != entry.get("hash"):
            problems.append(f"entry {i}: content changed after it was written")
        prev = entry.get("hash", "")
    return problems


# ---- the service ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortResult:
    phenotype: str
    version: int
    sha256: str
    attrition: list[dict[str, Any]]
    breakdown: list[dict[str, Any]]
    note: str = (
        "Counts from 1 to 10 are suppressed, and so are other cells where subtraction would "
        "reveal one. A suppressed cell can be any size: the label says nothing about it."
    )


class Refused(RuntimeError):
    """The service would not run the request; the reason is in the audit log too."""


@dataclass
class CohortService:
    engine: Engine
    tables: Mapping[str, str]
    approvals: Approvals
    audit: AuditLog
    source: str = "omop"

    def run(self, path: Path, requester: str, request: str = "") -> CohortResult:
        """Run an approved phenotype and return suppressed counts. Every outcome is audited."""
        who: dict[str, Any] = {"requester": requester, "request": request, "path": path.as_posix()}
        try:
            pheno = load(path)
        except (PhenotypeError, OSError, yaml.YAMLError) as exc:
            self.audit.append("refused", **who, reason=f"invalid phenotype: {exc}")
            raise Refused(f"invalid phenotype: {exc}") from exc
        who |= {"phenotype": pheno.name, "version": pheno.version, "sha256": pheno.sha256}
        status = self.approvals.status(pheno)
        if status != APPROVE:
            self.audit.append("refused", **who, reason=f"definition not approved ({status})")
            raise Refused(f"{pheno.name} v{pheno.version} is not approved ({status})")
        sql = compile_sql(pheno, self.tables)
        attrition = self.engine(sql["attrition"])
        shown = suppress_attrition(attrition["persons"])
        rows = [
            {"step": int(s), "criterion": str(c), "persons": p}
            for s, c, p in zip(attrition["step"], attrition["criterion"], shown, strict=True)
        ]
        breakdown: list[dict[str, Any]] = []
        if "breakdown" in sql:
            frame = self.engine(sql["breakdown"])
            final = int(attrition["persons"].iloc[-1])
            cells, rows[-1]["persons"] = suppress_breakdown(frame["persons"], final, shown[-1])
            keys = [c for c in frame.columns if c != "persons"]
            breakdown = [
                {**{k: (None if pd.isna(r[k]) else int(r[k])) for k in keys}, "persons": cell}
                for (_, r), cell in zip(frame.iterrows(), cells, strict=True)
            ]
        self.audit.append("ran", **who, source=self.source, steps=len(rows),
                          final=rows[-1]["persons"])  # fmt: skip
        return CohortResult(pheno.name, pheno.version, pheno.sha256, rows, breakdown)

    def decide(
        self, path: Path, decision: str, reviewer: str, reason: str, request: str = ""
    ) -> Decision:
        """Record a reviewer's decision on the exact definition in ``path``, and audit it."""
        pheno = load(path)
        made = Decision(
            pheno.name, pheno.version, pheno.sha256, decision, reviewer, reason, request
        )
        self.approvals.record(made)
        self.audit.append(
            "decision", phenotype=pheno.name, version=pheno.version, sha256=pheno.sha256,
            decision=decision, reviewer=reviewer, reason=reason, request=request,
        )  # fmt: skip
        return made


def duckdb_engine(tables: Mapping[str, pd.DataFrame]) -> tuple[Engine, dict[str, str]]:
    """An in-process engine over OMOP tables held as pandas frames (tests, local runs)."""
    import duckdb

    con = duckdb.connect()
    for name, frame in tables.items():
        con.register(name, frame)

    def run(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    return run, {t: t for t in OMOP_TABLES}
