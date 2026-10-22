"""Clinical text extraction (G5): label choices from the concept map, the keyword baseline,
the LLM path with a recorded transport (CI never calls an LLM), the budget guard, scoring,
and the rule that note text stays outside the repository. The notes here are made up."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cdm.extract import (
    HISTOPATHOLOGY,
    INVALID,
    NOT_STATED,
    OTHER,
    REPO,
    BudgetExceeded,
    MimicIVNotes,
    MTSamples,
    Note,
    SpendLog,
    choices,
    error_kind,
    keyword_extract,
    llm_extract,
    note_id,
    outside_repo,
    parse_reply,
    score,
    split_notes,
    write_sheet,
)
from cdm.omop import load_concept_map

ALLOWED = choices(load_concept_map(REPO / "config" / "omop_concepts.csv"))


def note(text: str) -> Note:
    return Note(note_id(text), "test", "Dermatology", "t", text)


BIOPSY = note(
    "A 54-year-old man with a changing pigmented lesion on the upper back. Shave biopsy "
    "showed melanoma in situ, margins clear."
)
NO_DX = note("Follow-up visit. Patient reports itching on the forearm. Plan: emollients.")
PSORIASIS = note("Psoriasis plaques on both elbows, stable on topical therapy.")


def test_choices_come_from_the_concept_map_with_concept_ids() -> None:
    assert ALLOWED.value("diagnosis", "Melanoma in situ") == "40391314"
    assert ALLOWED.value("body_site", "Posterior trunk") == "4280364"
    assert ALLOWED.value("confirmation", HISTOPATHOLOGY) == "43531286"
    # Labels sharing a concept score the same; anything else is invalid.
    same = ALLOWED.value("diagnosis", "Pigmented benign keratosis")
    assert same == ALLOWED.value("diagnosis", "Lichen planus like keratosis")
    assert ALLOWED.value("diagnosis", "melanoma (probably)") == INVALID
    assert ALLOWED.labels("diagnosis")[-2:] == [OTHER, NOT_STATED]
    assert "Melanoma metastasis" not in ALLOWED.labels("diagnosis")  # concept 0


def test_keyword_baseline_prefers_the_first_and_most_specific_mention() -> None:
    assert keyword_extract(BIOPSY, ALLOWED) == {
        "diagnosis": "Melanoma in situ",
        "body_site": "Posterior trunk",
        "confirmation": HISTOPATHOLOGY,
    }
    got = keyword_extract(NO_DX, ALLOWED)
    assert got["diagnosis"] == NOT_STATED and got["body_site"] == "Forearm"
    assert got["confirmation"] == NOT_STATED


def recorded(replies: dict[str, str]):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def send(system: str, user: str) -> tuple[str, dict[str, int]]:
        key = next(k for k in replies if k in user)
        calls.append(key)
        return replies[key], {"input_tokens": 1000, "output_tokens": 40}

    return send, calls


def test_llm_answers_are_parsed_cached_and_costed(tmp_path: Path) -> None:
    send, calls = recorded({
        "Shave biopsy": '{"diagnosis": "Melanoma in situ", "body_site": "Posterior trunk", '
                        '"confirmation": "histopathology"}',
        "Psoriasis": 'Here you go: {"diagnosis": "other", "body_site": "Elbow", '
                     '"confirmation": "not stated"}',
    })  # fmt: skip
    spend = SpendLog(tmp_path / "spend.jsonl")
    cache = tmp_path / "cache.jsonl"
    first = llm_extract(BIOPSY, ALLOWED, send, spend, cache)
    assert first["diagnosis"] == "Melanoma in situ"
    assert llm_extract(PSORIASIS, ALLOWED, send, spend, cache)["diagnosis"] == OTHER
    assert llm_extract(BIOPSY, ALLOWED, send, spend, cache) == first  # from the cache
    assert calls == ["Shave biopsy", "Psoriasis"]
    # Two calls of 1,000 input and 40 output tokens at $0.10 and $0.50 per million.
    assert spend.total() == pytest.approx(2 * (1000 * 0.10 + 40 * 0.50) / 1e6)
    entry = spend.entries()[0]
    assert set(entry) >= {"model", "note_id", "prompt_sha", "input_tokens", "cost_usd"}
    assert "text" not in json.dumps(entry)


def test_the_budget_guard_stops_a_call_before_the_cap(tmp_path: Path) -> None:
    send, calls = recorded({"Shave biopsy": "{}"})
    spend = SpendLog(tmp_path / "spend.jsonl", cap_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        llm_extract(BIOPSY, ALLOWED, send, spend, tmp_path / "cache.jsonl")
    assert calls == [] and spend.total() == 0


def test_unparseable_or_partial_replies_are_invalid() -> None:
    assert parse_reply("no json here") == {f: INVALID for f in
                                           ("diagnosis", "body_site", "confirmation")}  # fmt: skip
    got = parse_reply('{"diagnosis": "Nevus", "body_site": 3}')
    assert got == {"diagnosis": "Nevus", "body_site": INVALID, "confirmation": INVALID}


def test_errors_are_categorised() -> None:
    assert error_kind("diagnosis", "Melanoma in situ", "Melanoma in situ", ALLOWED) is None
    assert error_kind("diagnosis", "Melanoma in situ", "Melanoma, NOS", ALLOWED) == (
        "wrong granularity"
    )
    assert error_kind("diagnosis", "Nevus", NOT_STATED, ALLOWED) == "missed"
    assert error_kind("diagnosis", NOT_STATED, "Nevus", ALLOWED) == "hallucinated"
    assert error_kind("diagnosis", "Nevus", "Basal cell carcinoma", ALLOWED) == "wrong concept"
    assert error_kind("diagnosis", "Nevus", "naevus!", ALLOWED) == "invalid"


def test_score_reports_accuracy_stated_precision_recall_and_errors() -> None:
    labels = pd.DataFrame([
        {"note_id": "a", "label_diagnosis": "Nevus", "label_body_site": "Head",
         "label_confirmation": HISTOPATHOLOGY},
        {"note_id": "b", "label_diagnosis": NOT_STATED, "label_body_site": NOT_STATED,
         "label_confirmation": NOT_STATED},
    ])  # fmt: skip
    predictions = {
        "a": {"diagnosis": "Nevus", "body_site": "Neck", "confirmation": HISTOPATHOLOGY},
        "b": {"diagnosis": "Nevus", "body_site": NOT_STATED, "confirmation": NOT_STATED},
    }
    got = score(labels, predictions, ALLOWED)
    assert got["diagnosis"]["accuracy"] == 0.5
    assert got["diagnosis"]["stated_precision"] == 0.5 and got["diagnosis"]["stated_recall"] == 1
    assert got["diagnosis"]["errors"] == {"hallucinated": 1}
    assert got["body_site"]["errors"] == {"wrong granularity": 1}
    assert got["confirmation"]["accuracy"] == 1.0
    assert "text" not in json.dumps(got)


def test_note_text_never_lands_in_the_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        write_sheet([BIOPSY], REPO / "results" / "sheet", ALLOWED)
    assert outside_repo(tmp_path) == tmp_path
    sheet = write_sheet([BIOPSY, NO_DX, PSORIASIS], tmp_path, ALLOWED, n=3)
    rows = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    assert list(rows["split"]).count("dev") == 1 and len(rows) == 3  # at least one per group
    assert set(rows["label_diagnosis"]) == {""}
    legend = pd.read_csv(tmp_path / "labelling_choices.csv", dtype=str, keep_default_na=False)
    assert {"label_diagnosis", "label_body_site", "label_confirmation"} == set(legend["column"])


def test_splits_are_fixed_by_note_id_before_any_prompt_runs() -> None:
    notes = [note(f"Note number {i} about a nevus on the arm.") for i in range(10)]
    first = split_notes(notes, n=10)
    assert first == split_notes(list(reversed(notes)), n=10)
    assert [s for _, s in first].count("dev") == 1  # DEV_SHARE 0.1 of 10 notes


def test_mtsamples_keeps_dermatology_and_skin_pathology(tmp_path: Path) -> None:
    path = tmp_path / "mtsamples.csv"
    pd.DataFrame({
        "medical_specialty": [" Dermatology", " Lab Medicine - Pathology",
                              " Lab Medicine - Pathology", " Cardiovascular / Pulmonary"],
        "sample_name": ["a", "b", "c", "d"],
        "transcription": ["Nevus on the back.", "Skin, left arm, shave biopsy: nevus.",
                          "Breast, core biopsy: fibroadenoma.", "Chest pain."],
    }).to_csv(path, index=False)  # fmt: skip
    assert [n.title for n in MTSamples(path).notes()] == ["a", "b"]


def test_mimic_notes_need_the_tier_2_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "discharge.csv"
    pd.DataFrame({"note_id": ["n1", "n2"], "text": ["Melanoma on the scalp.", "Asthma."]}).to_csv(
        path, index=False
    )
    monkeypatch.delenv("CDM_TIER2", raising=False)
    with pytest.raises(PermissionError):
        list(MimicIVNotes(path).notes())
    monkeypatch.setenv("CDM_TIER2", "1")
    assert [n.note_id for n in MimicIVNotes(path).notes()] == ["n1"]


def test_an_invalid_answer_is_never_correct() -> None:
    assert error_kind("diagnosis", "not a label", "garbage", ALLOWED) == "invalid"


def test_a_failed_call_is_logged_at_its_worst_case_and_raised(tmp_path: Path) -> None:
    def broken(system: str, user: str) -> tuple[str, dict[str, int]]:
        raise TimeoutError("no reply")

    spend = SpendLog(tmp_path / "spend.jsonl")
    with pytest.raises(TimeoutError):
        llm_extract(BIOPSY, ALLOWED, broken, spend, tmp_path / "cache.jsonl")
    assert spend.entries()[0]["failed"] is True and spend.total() > 0


def test_mimic_notes_are_never_sent_to_an_outside_api(tmp_path: Path) -> None:
    send, calls = recorded({"x": "{}"})
    note = Note("n1", "mimic-iv-note", "", "", "x")
    with pytest.raises(PermissionError):
        llm_extract(note, ALLOWED, send, SpendLog(tmp_path / "s.jsonl"), tmp_path / "c.jsonl")
    assert calls == []


def test_an_existing_sheet_is_never_overwritten(tmp_path: Path) -> None:
    write_sheet([BIOPSY], tmp_path, ALLOWED)
    with pytest.raises(FileExistsError):
        write_sheet([NO_DX], tmp_path, ALLOWED)


def test_any_git_checkout_counts_as_the_repository(tmp_path: Path) -> None:
    (tmp_path / "other" / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the repository"):
        outside_repo(tmp_path / "other" / "notes")


def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    import importlib.util

    monkeypatch.setenv("CDM_NOTES_DIR", str(tmp_path / "work"))
    spec = importlib.util.spec_from_file_location("extraction", REPO / "scripts" / "extraction.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(module, "SPEND", tmp_path / "spend.jsonl")  # noqa: B010
    setattr(module, "OUT", tmp_path / "score.json")  # noqa: B010
    return module


def test_a_score_from_too_few_notes_or_another_prompt_is_not_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = cli(tmp_path, monkeypatch)
    work = tmp_path / "work"
    write_sheet([BIOPSY, NO_DX, PSORIASIS], work, ALLOWED, n=3)
    sheet = pd.read_csv(work / "labelling_sheet.csv", dtype=str, keep_default_na=False)
    sheet["label_diagnosis"], sheet["label_body_site"] = NOT_STATED, NOT_STATED
    sheet["label_confirmation"] = NOT_STATED
    sheet.to_csv(work / "labelling_sheet.csv", index=False)
    scoring = sheet[sheet["split"] == "scoring"]
    preds = scoring[["note_id"]].assign(diagnosis=NOT_STATED, body_site=NOT_STATED,
                                        confirmation=NOT_STATED, prompt_sha="old")  # fmt: skip
    preds.to_csv(work / "predictions_llm_scoring.csv", index=False)
    (work / "scoring_prompt_sha.txt").write_text("new")
    module.score_all()
    result = json.loads((tmp_path / "score.json").read_text())
    assert result["reportable"] is False
    assert len(result["not_reportable_because"]) == 2  # too few notes, and the wrong prompt
    assert result["methods"]["llm"]["diagnosis"]["accuracy"] == 1.0


def test_labels_outside_the_choices_stop_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = cli(tmp_path, monkeypatch)
    work = tmp_path / "work"
    write_sheet([BIOPSY, NO_DX], work, ALLOWED, n=2)
    sheet = pd.read_csv(work / "labelling_sheet.csv", dtype=str, keep_default_na=False)
    sheet["label_diagnosis"], sheet["label_body_site"] = "melanomaa", NOT_STATED
    sheet["label_confirmation"] = NOT_STATED
    sheet.to_csv(work / "labelling_sheet.csv", index=False)
    with pytest.raises(SystemExit, match="labels not in"):
        module.score_all()


def test_lesion_rules_take_lesion_notes_and_leave_incidental_skin() -> None:
    from cdm.extract import GENERIC, SPECIFIC

    assert SPECIFIC.search("Shave biopsy of a pigmented lesion on the back.")
    assert SPECIFIC.search("History of basal cell carcinomas of the nose.")
    assert SPECIFIC.search("Biopsy of the skin of the left arm.")
    assert GENERIC.search("SKIN: No skin lesions.") and not SPECIFIC.search(
        "SKIN: No skin lesions."
    )
    for text in ("The skin was closed with 4-0 nylon.", "Squamous cell carcinoma of the larynx."):
        assert not SPECIFIC.search(text) and not GENERIC.search(text)


def test_widened_selection_and_a_development_split_from_every_group(tmp_path: Path) -> None:
    from cdm.extract import MTSamples, selection_summary, stratum

    rows = [
        ("Dermatology", "derm", "Nevus on the arm, observed."),
        ("Surgery", "excision", "Excision of a basal cell carcinoma of the cheek."),
        ("General Medicine", "ros", "SKIN: No skin lesions. HEENT: normal."),
        ("Surgery", "closure", "The skin was closed with staples."),
        ("Surgery", "dup", "Nevus on the arm, observed."),  # same text as the Dermatology note
    ]
    csv_path = tmp_path / "m.csv"
    pd.DataFrame(rows, columns=["medical_specialty", "sample_name", "transcription"]).to_csv(
        csv_path, index=False
    )
    notes = list(MTSamples(csv_path).notes())
    assert len(notes) == 4  # the closure note is left out; the duplicate text is kept once below
    split = split_notes(notes)
    assert len(split) == 3 and {stratum(x) for x, _ in split} == {
        "dermatology",
        "specific_mention",
        "generic_mention_only",
    }
    assert [s for _, s in split].count("dev") == 3  # at least one from each group
    summary = selection_summary(notes)
    assert summary["by_group_and_split"]["dermatology"] == {"dev": 1, "scoring": 0}
    assert "text" not in json.dumps(summary)
