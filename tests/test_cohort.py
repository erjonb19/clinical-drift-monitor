"""The governed cohort service: allowlist, approval by hash, suppression, audit, and counts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdm.cohort import (
    COMPLEMENT,
    SUPPRESSED,
    Approvals,
    AuditLog,
    CohortService,
    PhenotypeError,
    Refused,
    Release,
    Scope,
    compile_sql,
    duckdb_engine,
    guard_releases,
    load,
    parse,
    suppress,
    suppress_attrition,
    suppress_breakdown,
    verify_chain,
)
from cdm.omop import build_tables, load_concept_map

REPO = Path(__file__).resolve().parents[1]
CMAP = load_concept_map(REPO / "config" / "omop_concepts.csv")
MEL, MEL_INSITU, NEVUS, HISTO = 4162276, 40391314, 4271013, 43531286

BASE = {"name": "mel", "version": 1, "title": "t",
        "index": {"condition_concept_ids": [MEL, MEL_INSITU]}}  # fmt: skip


# ---- the allowlisted language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sql": "SELECT * FROM person"}, "unknown keys"),
        ({"index": {"condition_concept_ids": ["4162276; DROP TABLE person"]}}, "positive integers"),
        ({"index": {"condition_concept_ids": [MEL], "age_min": "18 OR 1=1"}}, "integer age"),
        ({"include": [{"person_source_value": [1]}]}, "not an allowed criterion"),
        (
            {"include": [{"procedure_concept_ids": [HISTO], "gender_concept_ids": [1]}]},
            "exactly one",
        ),
        ({"breakdown": ["person_source_value"]}, "breakdown"),
        ({"version": 0}, "positive integer"),
    ],
)
def test_anything_outside_the_allowlist_is_refused(change: dict[str, object], message: str) -> None:
    with pytest.raises(PhenotypeError, match=message):
        parse(BASE | change)


def test_hash_follows_meaning_not_formatting(tmp_path: Path) -> None:
    folder = tmp_path / "mel"
    folder.mkdir()
    (folder / "v1.yaml").write_text(
        "name: mel\nversion: 1\ntitle: t\nindex: {condition_concept_ids: [40391314, 4162276]}\n"
    )
    a = load(folder / "v1.yaml")
    (folder / "v1.yaml").write_text(
        "# a comment\nname: mel\nversion: 1\ntitle: t\nindex:\n"
        "  condition_concept_ids: [4162276, 40391314]\n"
    )
    assert load(folder / "v1.yaml").sha256 == a.sha256
    assert parse(BASE | {"include": [{"procedure_concept_ids": [HISTO]}]}).sha256 != a.sha256
    (folder / "v2.yaml").write_text((folder / "v1.yaml").read_text())  # says version 1
    with pytest.raises(PhenotypeError, match="expected mel/v1.yaml"):
        load(folder / "v2.yaml")


def test_committed_phenotypes_are_valid_and_use_mapped_concepts() -> None:
    mapped = {int(c) for c in CMAP["concept_id"] if c != 0}
    files = sorted((REPO / "phenotypes").glob("*/v*.yaml"))
    assert len(files) >= 3
    for path in files:
        d = load(path).definition
        ids = list(d["index"]["condition_concept_ids"])
        for criterion in d["include"] + d["exclude"]:
            ids += next(iter(criterion.values()))
        assert set(ids) <= mapped, path


# ---- small cells -----------------------------------------------------------------------------


def test_small_cells_and_the_cell_that_would_reveal_one_are_hidden() -> None:
    assert suppress(pd.Series([5, 50, 30]), total=85) == [SUPPRESSED] * 3
    assert suppress(pd.Series([5, 7, 30]), total=42) == [SUPPRESSED] * 3
    assert suppress(pd.Series([0, 50, 12]), total=62) == ["0", "50", "12"]
    assert suppress(pd.Series([5, 50])) == [SUPPRESSED, "50"]  # no total published
    # 100 -> 95 drops 5 people: 95 is hidden too; 95 -> 40 drops 55, so 40 is shown.
    assert suppress_attrition(pd.Series([100, 95, 40, 3])) == ["100", COMPLEMENT, "40", SUPPRESSED]
    # Eleven people in eleven cells of one: hiding the cells is not enough, the total goes too.
    cells, total = suppress_breakdown(pd.Series([1] * 11 + [0]), 11, "11")
    assert cells == [SUPPRESSED] * 11 + ["0"] and total == SUPPRESSED
    # A step after a hidden one: its drop counts from the last shown step (100 -> 95 is 5).
    assert suppress_attrition(pd.Series([100, 95, 95, 40])) == ["100", COMPLEMENT, COMPLEMENT,
                                                                "40"]  # fmt: skip


def _pinned(outputs: dict[tuple[str, ...], list[tuple[int, ...]]]) -> list[object]:
    """Outputs from which some count from 1 to 10 is the same in every true vector that
    could have produced them, so a reader would know it."""
    leaks: list[object] = []
    for shown, vectors in outputs.items():
        for i in range(len(vectors[0])):
            values = {v[i] for v in vectors}
            if len(values) == 1 and 0 < next(iter(values)) < 11:
                leaks.append((shown, i, values))
    return leaks


def test_no_small_count_can_be_worked_out_from_any_output() -> None:
    """Brute force: every breakdown of three cells (0 to 24 each) under its published total,
    and every three-step attrition (with its drops), grouped by what is shown."""
    from itertools import product

    cells: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
    for v in product(range(25), repeat=3):
        cells_shown, total = suppress_breakdown(
            pd.Series(v), sum(v), suppress_attrition(pd.Series([sum(v)]))[0]
        )
        shown = (*cells_shown, total)
        cells.setdefault(shown, []).append(v)
    assert _pinned(cells) == []
    steps: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
    for a, b, c in product(range(40), repeat=3):
        if a >= b >= c:
            shown = tuple(suppress_attrition(pd.Series([a, b, c])))
            steps.setdefault(shown, []).append((a, b, c, a - b, b - c, a - c))
    assert _pinned(steps) == []


def test_a_hidden_final_count_hides_the_breakdown(tmp_path: Path) -> None:
    frame = silver(40)
    nevi = frame.index[frame["label"] == "nv"]  # 30 people
    frame.loc[nevi, "ham_dx_type"] = "follow_up"
    frame.loc[nevi[:5], "ham_dx_type"] = "histo"
    engine, tables = duckdb_engine(omop(frame))
    svc = CohortService(engine, tables, Approvals(tmp_path / "approvals.jsonl"),
                        AuditLog(tmp_path / "audit.jsonl"))  # fmt: skip
    definition = {"name": "nv", "version": 1, "title": "t",
                  "index": {"condition_concept_ids": [NEVUS]},
                  "exclude": [{"procedure_concept_ids": [HISTO]}],
                  "breakdown": ["gender_concept_id"]}  # fmt: skip
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    result = svc.run(path, "fellow")
    # 30 -> 25 drops 5 people, so 25 is hidden, and so is the one cell that would equal it.
    assert [r["persons"] for r in result.attrition] == ["30", COMPLEMENT]
    assert [r["persons"] for r in result.breakdown] == [COMPLEMENT]


def test_bad_versions_and_age_ranges_are_refused() -> None:
    with pytest.raises(PhenotypeError, match="positive integer"):
        parse(BASE | {"version": True})
    with pytest.raises(PhenotypeError, match="age_min is above age_max"):
        parse(BASE | {"index": {"condition_concept_ids": [MEL], "age_min": 50, "age_max": 20}})


# ---- approval, audit and counts on real OMOP tables (built from synthetic silver) ----------


def silver(n_people: int = 40) -> pd.DataFrame:
    """Lesion-level people: every 4th has melanoma (in situ every 8th), histology on evens."""
    rows = []
    for i in range(n_people):
        mel = i % 4 == 0
        rows.append({
            "isic_id": f"I{i}", "source": "ham10000", "lesion_id": f"L{i}", "patient_id": None,
            "sex": "male", "age_approx": 15.0 + i, "image_type": "dermoscopic",
            "label": "mel" if mel else "nv",
            "diagnosis_3": ("Melanoma in situ" if i % 8 == 0 else "Melanoma, NOS") if mel
            else "Nevus",
            "ham_dx_type": "histo" if i % 2 == 0 else "follow_up", "ham_localization": "back",
            "license": "CC-BY-NC", "attribution": "x", "local_path": None,
        })  # fmt: skip
    return pd.DataFrame(rows)


def omop(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    concepts = pd.DataFrame(
        {"concept_id": sorted({str(c) for c in CMAP["concept_id"] if c != 0})}
    ).assign(concept_name="x", domain_id="x", vocabulary_id="x", concept_class_id="x",
             standard_concept="S", concept_code="x", valid_start_date="19700101",
             valid_end_date="20991231", invalid_reason="")  # fmt: skip
    tables = build_tables(frame, CMAP, concepts, "t", "v", date(2026, 1, 1))
    # DuckDB reads typed columns; the pipeline's object columns become typed in Spark.
    return {k: v.infer_objects() for k, v in tables.items()}


def make_service(tmp_path: Path, n_people: int = 100) -> CohortService:
    engine, tables = duckdb_engine(omop(silver(n_people)))
    return CohortService(engine, tables, Approvals(tmp_path / "approvals.jsonl"),
                         AuditLog(tmp_path / "audit.jsonl"))  # fmt: skip


def write(tmp_path: Path, definition: dict[str, object]) -> Path:
    import yaml

    folder = tmp_path / str(definition["name"])
    folder.mkdir(exist_ok=True)
    path = folder / f"v{definition['version']}.yaml"
    path.write_text(yaml.safe_dump(definition))
    return path


def test_only_an_approved_exact_definition_runs_and_everything_is_audited(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    path = write(tmp_path, BASE)
    with pytest.raises(Refused, match="pending"):
        svc.run(path, "fellow")
    svc.decide(path, "escalate", "data engineer", "needs the PI")
    with pytest.raises(Refused, match="pending"):  # escalation is not a verdict
        svc.run(path, "fellow")
    svc.decide(path, "approve", "PI", "public data, counts only")
    assert svc.run(path, "fellow").attrition[0]["persons"] == "25"  # i = 0, 4, ..., 96
    # Changing the definition after approval needs a new approval.
    write(tmp_path, BASE | {"index": {"condition_concept_ids": [MEL]}})
    with pytest.raises(Refused, match="pending"):
        svc.run(path, "fellow")
    write(tmp_path, BASE)
    svc.decide(path, "reject", "PI", "withdrawn")
    with pytest.raises(Refused, match="reject"):
        svc.run(path, "fellow")
    events = [e["event"] for e in svc.audit.entries()]
    assert events == ["refused", "decision", "refused", "decision", "ran", "refused", "decision",
                      "refused"]  # fmt: skip
    assert verify_chain(svc.audit.entries()) == []


def test_counts_match_the_data_and_small_cells_are_suppressed(tmp_path: Path) -> None:
    svc = make_service(tmp_path, n_people=40)
    definition = BASE | {
        "index": {"condition_concept_ids": [MEL, MEL_INSITU], "age_min": 30},
        "include": [{"procedure_concept_ids": [HISTO]}],
        "breakdown": ["age_band"],
    }
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    result = svc.run(path, "fellow")
    # Melanoma people are i = 0, 4, ..., 36 (age 15 + i); aged 30+ means i >= 16: 16..36,
    # six people, all even so all with histology. Six is a small cell.
    assert [r["persons"] for r in result.attrition] == [SUPPRESSED, SUPPRESSED]
    # Ages 31, 35, 39, 43, 47, 51: bands 30 (3), 40 (2), 50 (1), all small.
    assert {r["age_band"]: r["persons"] for r in result.breakdown} == {
        30: SUPPRESSED, 40: SUPPRESSED, 50: SUPPRESSED}  # fmt: skip
    # The SQL itself returns the exact counts the suppression hides.
    raw = svc.engine(compile_sql(load(path), svc.tables)["attrition"])
    assert raw["persons"].tolist() == [6, 6]


def test_exclusion_and_a_larger_cohort_show_unsuppressed_counts(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    definition = {"name": "nv", "version": 1, "title": "t",
                  "index": {"condition_concept_ids": [NEVUS]},
                  "exclude": [{"procedure_concept_ids": [HISTO]}]}  # fmt: skip
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    result = svc.run(path, "fellow")
    # 75 nevus people (i % 4 in 1, 2, 3); the 25 with i % 4 == 2 had histology.
    assert [r["persons"] for r in result.attrition] == ["75", "50"]


def test_audit_chain_detects_an_edited_or_deleted_line(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append("ran", phenotype=f"p{i}")
    lines = log.path.read_text().splitlines()
    edited = json.loads(lines[1]) | {"phenotype": "other"}
    entries = [json.loads(lines[0]), edited, json.loads(lines[2])]
    assert any("changed" in p for p in verify_chain(entries))
    assert any("does not follow" in p for p in verify_chain([json.loads(lines[0]),
                                                             json.loads(lines[2])]))  # fmt: skip


# ---- across requests -------------------------------------------------------------------------


def people(groups: list[tuple[int, str, int, bool]]) -> pd.DataFrame:
    """Melanoma people from (how many, sex, age, has histology) groups."""
    frame = silver(sum(n for n, *_ in groups))
    i = 0
    for n, sex, age, histo in groups:
        frame.loc[i : i + n - 1, ["sex", "age_approx", "label", "diagnosis_3", "ham_dx_type"]] = [
            sex, float(age), "mel", "Melanoma, NOS", "histo" if histo else "follow_up"]  # fmt: skip
        i += n
    return frame


def service_on(tmp_path: Path, frame: pd.DataFrame) -> CohortService:
    engine, tables = duckdb_engine(omop(frame))
    return CohortService(engine, tables, Approvals(tmp_path / "approvals.jsonl"),
                         AuditLog(tmp_path / "audit.jsonl"))  # fmt: skip


def approved(svc: CohortService, tmp_path: Path, definition: dict[str, object]) -> Path:
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    return path


def test_a_nested_result_cannot_reveal_a_small_group_by_age(tmp_path: Path) -> None:
    """Each result is safe on its own (34 people; 30 adults), but together they say four
    people with melanoma are under 18. The second count is hidden."""
    svc = service_on(tmp_path, people([(30, "female", 45, True), (4, "male", 15, True)]))
    first = svc.run(approved(svc, tmp_path, BASE | {"name": "everyone"}), "fellow")
    assert first.attrition[-1]["persons"] == "34"
    adults = BASE | {
        "name": "adults",
        "index": {"condition_concept_ids": [MEL, MEL_INSITU], "age_min": 18},
    }
    second = svc.run(approved(svc, tmp_path, adults), "coordinator")
    assert second.attrition[-1]["persons"] == SUPPRESSED
    assert svc.audit.entries()[-1]["hidden_because_of"] == ["everyone v1"]


def test_a_nested_total_that_differs_by_a_few_people_is_hidden(tmp_path: Path) -> None:
    svc = service_on(tmp_path, people([(30, "female", 45, True), (4, "female", 45, False)]))
    total = svc.run(approved(svc, tmp_path, BASE | {"name": "all"}), "fellow")
    assert total.attrition[-1]["persons"] == "34"
    histo = BASE | {"name": "histo", "include": [{"procedure_concept_ids": [HISTO]}],
                    "breakdown": ["gender_concept_id"]}  # fmt: skip
    result = svc.run(approved(svc, tmp_path, histo), "fellow")
    # 34 - 30 would be the 4 people without histology: the 30 and its breakdown are hidden.
    assert [r["persons"] for r in result.attrition] == ["34", COMPLEMENT]
    assert all(r["persons"] == COMPLEMENT for r in result.breakdown)


def test_unrelated_and_repeated_requests_are_not_suppressed_further(tmp_path: Path) -> None:
    svc = service_on(tmp_path, people([(30, "female", 45, True), (20, "male", 45, False)]))
    path = approved(svc, tmp_path, BASE | {"name": "all", "breakdown": ["gender_concept_id"]})
    first = svc.run(path, "fellow")
    assert svc.run(path, "student") == first  # the same definition again
    women = BASE | {"name": "women", "include": [{"gender_concept_ids": [8532]}]}
    # 50 people, 30 of them women: a difference of 20 discloses nothing.
    assert svc.run(approved(svc, tmp_path, women), "fellow").attrition[-1]["persons"] == "30"


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "cohort.py"), *args],
                          capture_output=True, text=True, cwd=REPO)  # fmt: skip


def test_stand_in_approvals_stay_apart_and_serve_only_synthetic_requests(tmp_path: Path) -> None:
    library = REPO / "phenotypes" / "approvals.jsonl"
    before = library.read_bytes() if library.exists() else None
    files = ["--audit", str(tmp_path / "audit.jsonl"), "--approvals", str(tmp_path / "a.jsonl")]
    pheno = str(REPO / "phenotypes" / "melanoma_any" / "v1.yaml")
    decided = cli(*files, "decide", pheno, "approve", "--reviewer", "stand-in", "--reason", "x")
    assert decided.returncode == 0, decided.stderr
    assert (tmp_path / "a.jsonl").read_text().count("approve") == 1
    assert (library.read_bytes() if library.exists() else None) == before
    real = cli(*files, "run", pheno, "--requester", "fellow", "--data", str(tmp_path))
    assert real.returncode == 2 and "only serves synthetic" in real.stderr


def scope(conditions: set[int], low: int = 0, high: int = 200, inc: set[str] | None = None,
          exc: set[str] | None = None) -> Scope:  # fmt: skip
    return Scope(frozenset(conditions), low, high, frozenset(inc or ()), frozenset(exc or ()))


def test_nesting_follows_conditions_age_ranges_and_criteria() -> None:
    assert scope({1}, 18).within(scope({1, 2}))  # fewer conditions, narrower ages
    assert not scope({1, 2}).within(scope({1}, 18))
    assert scope({1}, inc={"a", "b"}).within(scope({1}, inc={"a"}))
    assert not scope({1}, inc={"a"}).within(scope({1}, inc={"a", "b"}))
    assert scope({1}, exc={"x"}).within(scope({1}))
    assert not scope({1}, 0, 50).within(scope({1}, 18, 200))


def release(s: Scope, total: int, cells: dict[int, int] | None, label: str) -> list[Release]:
    """A total and an optional sex breakdown, suppressed on their own as the service does."""
    step = Release(s, (), {(): total}, {(): suppress_attrition(pd.Series([total]))[0]}, label)
    if cells is None:
        return [step]
    keys = sorted(cells)
    shown, step.shown[()] = suppress_breakdown(
        pd.Series([cells[k] for k in keys]), total, step.shown[()]
    )
    table = Release(s, ("gender_concept_id",), {(k,): cells[k] for k in keys},
                    {(k,): v for k, v in zip(keys, shown, strict=True)}, label)  # fmt: skip
    return [step, table]


def test_one_hidden_row_under_a_shown_total_hides_the_total() -> None:
    """Review case: earlier 80 = 50 men + 30 women; now 45, all men, hidden by the guard.
    Shown, the 45 would be the men's count, and 50 - 45 = 5."""
    old = release(scope({1}), 80, {8507: 50, 8532: 30}, "old")
    new = release(scope({1}, inc={"x"}), 45, {8507: 45}, "new")
    guard_releases(new, old)
    assert new[0].shown[()] == SUPPRESSED


def test_fewer_conditions_with_an_age_limit_is_still_nested() -> None:
    old = release(scope({1, 2}), 100, None, "old")
    new = release(scope({1}, 18), 95, None, "new")
    assert guard_releases(new, old) == ["old"] and new[0].shown[()] == SUPPRESSED


def test_no_small_difference_can_be_worked_out_from_two_nested_releases() -> None:
    """Brute force: an earlier cohort (men, women up to 20 each) and a nested one inside it,
    both with a sex breakdown, released through the guard. Grouped by everything shown, no
    count or difference from 1 to 10 may be the same in every case behind one output."""
    from itertools import product

    seen: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
    for a, b in product(range(21), repeat=2):
        old = release(scope({1}), a + b, {8507: a, 8532: b}, "old")
        old_shown = tuple(v for r in old for v in r.shown.values())
        for c, d in product(range(a + 1), range(b + 1)):
            new = release(scope({1}, inc={"x"}), c + d, {8507: c, 8532: d}, "new")
            guard_releases(new, old)
            shown = old_shown + ("|",) + tuple(v for r in new for v in r.shown.values())
            seen.setdefault(shown, []).append((a, b, c, d, a - c, b - d, a + b - c - d))
    leaks = [
        (shown, i)
        for shown, cases in seen.items()
        for i in range(7)
        if len({x[i] for x in cases}) == 1
        and 0 < cases[0][i] < 11
        # The search stops at 20 per cell, which a reader does not know: skip cases where a
        # cell sits at that edge, since there the "pin" is the search's, not the data's.
        and all(max(x[:2]) < 20 for x in cases)
    ]
    assert leaks == []
