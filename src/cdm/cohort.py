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
4. **suppresses small cells**: a count from 1 to 10 is shown as "<11", and a second cell is
   suppressed where the suppressed one could otherwise be recovered by subtraction;
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
SUPPRESSED = "<11"
# A cell of 11 or more hidden only so a "<11" cell next to it cannot be recovered.
COMPLEMENT = "suppressed"
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
    if not isinstance(definition["version"], int) or definition["version"] < 1:
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
    """Counts as strings with cells from 1 to 10 shown as "<11". If exactly one cell is
    hidden and the total is published, the next smallest cell is hidden too ("suppressed"),
    so the first cannot be recovered by subtraction. Zero is shown: it reveals no person."""
    values = [int(v) for v in counts]
    small = {i for i, v in enumerate(values) if 0 < v < MIN_CELL}
    extra: set[int] = set()
    if total is not None and len(small) == 1:
        rest = [i for i, v in enumerate(values) if i not in small and v > 0]
        if rest:
            extra.add(min(rest, key=lambda i: values[i]))
    return [
        SUPPRESSED if i in small else COMPLEMENT if i in extra else str(v)
        for i, v in enumerate(values)
    ]


def suppress_attrition(persons: pd.Series) -> list[str]:
    """Attrition counts: a step from 1 to 10 shows "<11". A step's drop from the step before
    is also a count of people, so a step whose drop is from 1 to 10 is hidden too
    ("suppressed")."""
    values = [int(v) for v in persons]
    out = []
    for i, v in enumerate(values):
        drop = values[i - 1] - v if i else 0
        if 0 < v < MIN_CELL:
            out.append(SUPPRESSED)
        elif 0 < drop < MIN_CELL:
            out.append(COMPLEMENT)
        else:
            out.append(str(v))
    return out


# ---- across requests ------------------------------------------------------------------------
# Each result is suppressed on its own, but two released results whose cohorts are nested can
# still give a person away: their difference may be a group of 1 to 10, or one may pin down a
# cell the other hid. So before a result is released, every number in it is compared with the
# numbers already released for nested definitions, and the groups that would disclose are
# hidden too. Only nesting that the definitions show is detected (same index conditions, age
# range within, criteria a superset); overlap of any other kind is still the reviewer's check.

AGE_CEILING = 200


@dataclass(frozen=True)
class Scope:
    """The people one count covers: index conditions, an age range, and criteria."""

    conditions: frozenset[int]
    age_min: int
    age_max: int
    include: frozenset[str]
    exclude: frozenset[str]

    def within(self, other: Scope) -> bool:
        """Whether every person in this scope is in ``other`` too, as the definitions show."""
        if self.conditions != other.conditions:
            # Age is measured at the earliest index condition, so with an age limit a smaller
            # condition set can give a different age; nesting is only clear without limits.
            no_limits = {self.age_min, other.age_min} == {0} and {
                self.age_max, other.age_max} == {AGE_CEILING}  # fmt: skip
            if not (no_limits and self.conditions <= other.conditions):
                return False
        return (
            other.age_min <= self.age_min
            and self.age_max <= other.age_max
            and other.include <= self.include
            and other.exclude <= self.exclude
        )


def scopes(pheno: Phenotype) -> list[Scope]:
    """The scope of each attrition step: the index, then one more criterion per step."""
    d = pheno.definition
    steps = [("include", c) for c in d["include"]] + [("exclude", c) for c in d["exclude"]]
    out = []
    for i in range(len(steps) + 1):
        out.append(
            Scope(
                frozenset(d["index"]["condition_concept_ids"]),
                int(d["index"].get("age_min", 0)),
                int(d["index"].get("age_max", AGE_CEILING)),
                frozenset(canonical(c) for kind, c in steps[:i] if kind == "include"),
                frozenset(canonical(c) for kind, c in steps[:i] if kind == "exclude"),
            )
        )
    return out


Cell = tuple[int | None, ...]


@dataclass
class Release:
    """One released table (an attrition step, or a breakdown): true counts and what was
    shown, by cell. An attrition step is a table with no keys and one cell."""

    scope: Scope
    keys: tuple[str, ...]
    true: dict[Cell, int]
    shown: dict[Cell, str]
    label: str = ""
    criterion: str = ""


def _groups(release: Release, keys: tuple[str, ...]) -> dict[Cell, tuple[int, bool]]:
    """Per group of ``keys``: the true count, and whether the shown cells give it exactly."""
    where = [release.keys.index(k) for k in keys]
    out: dict[Cell, tuple[int, bool]] = {}
    for cell, value in release.true.items():
        group = tuple(cell[i] for i in where)
        total, exact = out.get(group, (0, True))
        out[group] = (total + value, exact and release.shown[cell].isdigit())
    return out


def disclosing_cells(new: Release, old: Release) -> set[Cell]:
    """Cells of ``new`` to hide because ``old`` was released: a group where the two differ by
    1 to 10 people, or where they are equal and one shows exactly what the other hid."""
    if not (new.scope.within(old.scope) or old.scope.within(new.scope)):
        return set()
    if set(new.keys) <= set(old.keys):
        keys = new.keys
    elif set(old.keys) <= set(new.keys):
        keys = old.keys
    else:
        return set()
    mine, theirs = _groups(new, keys), _groups(old, keys)
    hide = set()
    for group in mine.keys() | theirs.keys():
        a, a_exact = mine.get(group, (0, True))
        b, b_exact = theirs.get(group, (0, True))
        if 0 < abs(a - b) < MIN_CELL or (a == b and a_exact != b_exact):
            hide.add(group)
    where = [new.keys.index(k) for k in keys]
    return {cell for cell in new.true if tuple(cell[i] for i in where) in hide}


def guard_releases(new: list[Release], old: list[Release]) -> list[str]:
    """Hide cells of ``new`` (in place) that earlier releases would disclose, then keep the
    new result consistent: a hidden total hides the breakdown, and a breakdown with one
    hidden cell under a shown total hides one more. Returns the earlier releases that
    caused a cell to be hidden."""
    causes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for release in new:
            for earlier in old:
                for cell in disclosing_cells(release, earlier):
                    if release.shown[cell].isdigit():
                        release.shown[cell] = COMPLEMENT
                        causes.add(earlier.label)
                        changed = True
        final = [r for r in new if not r.keys][-1]
        for table in (r for r in new if r.keys):
            visible = [c for c, s in table.shown.items() if s.isdigit()]
            hidden = len(table.shown) - len(visible)
            if not final.shown[()].isdigit() and visible:
                for cell in visible:
                    table.shown[cell] = COMPLEMENT
                changed = True
            elif hidden == 1 and [c for c in visible if table.true[c] > 0]:
                cell = min((c for c in visible if table.true[c] > 0), key=table.true.__getitem__)
                table.shown[cell] = COMPLEMENT
                changed = True
    return sorted(causes)


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
    or deleted line breaks :func:`verify_chain`."""

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
        "Counts from 1 to 10 show as <11; a cell marked suppressed is hidden so that a small "
        "count cannot be worked out by subtraction, within this result or against an earlier "
        "result for a nested definition."
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
        steps, table = self._releases(pheno)
        causes = guard_releases(steps + ([table] if table else []), self._released(pheno))
        rows = [
            {"step": i, "criterion": r.criterion, "persons": r.shown[()]}
            for i, r in enumerate(steps)
        ]
        breakdown: list[dict[str, Any]] = []
        if table:
            breakdown = [
                {**dict(zip(table.keys, cell, strict=True)), "persons": shown}
                for cell, shown in table.shown.items()
            ]
        self.audit.append("ran", **who, source=self.source, steps=len(rows),
                          final=rows[-1]["persons"], definition=pheno.definition,
                          released={"attrition": rows, "breakdown": breakdown},
                          hidden_because_of=causes)  # fmt: skip
        return CohortResult(pheno.name, pheno.version, pheno.sha256, rows, breakdown)

    def _releases(self, pheno: Phenotype) -> tuple[list[Release], Release | None]:
        """True counts of a phenotype, suppressed on their own: one release per attrition
        step, and the breakdown if it has one."""
        sql = compile_sql(pheno, self.tables)
        attrition = self.engine(sql["attrition"])
        label = f"{pheno.name} v{pheno.version}"
        steps = [
            Release(scope, (), {(): int(v)}, {(): shown}, label, str(criterion))
            for scope, v, shown, criterion in zip(
                scopes(pheno), attrition["persons"], suppress_attrition(attrition["persons"]),
                attrition["criterion"], strict=True,
            )
        ]  # fmt: skip
        if "breakdown" not in sql:
            return steps, None
        frame = self.engine(sql["breakdown"])
        keys = tuple(pheno.definition["breakdown"])
        cells = [tuple(None if pd.isna(r[k]) else int(r[k]) for k in keys)
                 for _, r in frame.iterrows()]  # fmt: skip
        shown = suppress(frame["persons"], total=int(attrition["persons"].iloc[-1]))
        true = {c: int(v) for c, v in zip(cells, frame["persons"], strict=True)}
        return steps, Release(steps[-1].scope, keys, true, dict(zip(cells, shown, strict=True)),
                              label)  # fmt: skip

    def _released(self, pheno: Phenotype) -> list[Release]:
        """Everything already released for other definitions, as shown then, with true counts
        recomputed from the data."""
        out: list[Release] = []
        cache: dict[str, tuple[list[Release], Release | None]] = {}
        for entry in self.audit.entries():
            if entry["event"] != "ran" or "released" not in entry:
                continue
            if entry["sha256"] == pheno.sha256:
                continue
            if entry["sha256"] not in cache:
                cache[entry["sha256"]] = self._releases(parse(entry["definition"]))
            steps, table = cache[entry["sha256"]]
            shown = entry["released"]
            for step, row in zip(steps, shown["attrition"], strict=True):
                out.append(Release(step.scope, (), step.true, {(): row["persons"]}, step.label))
            if table:
                then = {tuple(r[k] for k in table.keys): r["persons"] for r in shown["breakdown"]}
                seen = {c: then.get(c, "0") for c in table.true}
                out.append(Release(table.scope, table.keys, table.true, seen, table.label))
        return out

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
