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
    assert list(rows["split"]).count("dev") == 1 and len(rows) == 3
    assert set(rows["label_diagnosis"]) == {""}
    legend = pd.read_csv(tmp_path / "labelling_choices.csv", dtype=str, keep_default_na=False)
    assert {"label_diagnosis", "label_body_site", "label_confirmation"} == set(legend["column"])


def test_splits_are_fixed_by_note_id_before_any_prompt_runs() -> None:
    notes = [note(f"Note number {i} about a nevus on the arm.") for i in range(10)]
    first = split_notes(notes, n=10)
    assert first == split_notes(list(reversed(notes)), n=10)
    assert [s for _, s in first].count("dev") == 2


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
