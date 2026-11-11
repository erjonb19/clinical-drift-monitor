"""The public demo's logic (G6), on a made-up stand-in for the ISIC response: visitors get a
scratch approvals file, their own definitions are versioned, and output is suppressed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cdm.cohort import Refused
from cdm.demo import DemoSession, custom_definition, diagnosis_options, fetch_tables, library


def records(n: int = 60) -> list[dict[str, Any]]:
    diagnoses = ["Nevus", "Basal cell carcinoma", "Melanoma, NOS", "Seborrheic keratosis"]
    return [
        {"isic_id": f"ISIC_SYN{i:04d}", "copyright_license": "CC-BY", "attribution": "synthetic",
         "files": {"full": {"url": ""}},
         "metadata": {"acquisition": {"image_type": "clinical: close-up"},
                      "clinical": {"patient_id": f"IP_{i // 2}", "lesion_id": f"IL_{i}",
                                   "sex": ["male", "female"][i % 2], "age_approx": 30 + i % 50,
                                   "anatom_site_1": "Head and neck", "anatom_site_2": "Head",
                                   "diagnosis_3": diagnoses[i % 4],
                                   "diagnosis_confirm_type": "histopathology"}}}
        for i in range(n)
    ]  # fmt: skip


@pytest.fixture
def session(tmp_path: Path) -> DemoSession:
    return DemoSession(fetch_tables(lambda collection: records()), folder=tmp_path)


def test_a_library_phenotype_needs_the_visitors_approval(session: DemoSession) -> None:
    path = next(p for p in library() if p.parent.name == "melanoma_any")
    with pytest.raises(Refused, match="not approved"):
        session.run(path)
    session.approve(path)
    result = session.run(path)
    assert result.attrition[0]["persons"] == "15"  # every 4th of 60
    assert list(session.audit()["event"]) == ["refused", "decision", "ran"]
    # The visitor's decision stays in the scratch folder, never in the library's file.
    assert (session.folder / "approvals.jsonl").exists()


def test_the_library_approvals_file_is_never_touched(session: DemoSession) -> None:
    from cdm.demo import REPO

    library_file = REPO / "phenotypes" / "approvals.jsonl"
    before = library_file.read_bytes() if library_file.exists() else None
    for path in library():
        session.approve(path)
        session.run(path)
    assert (library_file.read_bytes() if library_file.exists() else None) == before


def test_small_counts_come_out_suppressed(tmp_path: Path) -> None:
    session = DemoSession(fetch_tables(lambda collection: records(24)), folder=tmp_path)
    path = next(p for p in library() if p.parent.name == "melanoma_any")
    session.approve(path)
    result = session.run(path)
    assert result.attrition[0]["persons"] == "suppressed"  # 6 people with melanoma
    assert all(r["persons"] in ("suppressed", "0") for r in result.breakdown)


def test_concepts_are_ids_with_athena_links() -> None:
    from cdm.demo import concepts_in

    table = concepts_in(custom_definition(["Melanoma, NOS"], None, None, True, []))
    assert table["concept_id"].tolist() == [4162276, 43531286]
    assert table["name (Athena)"].str.startswith("https://athena.ohdsi.org/").all()


def test_a_changed_custom_definition_is_a_new_version(session: DemoSession) -> None:
    options = diagnosis_options()
    assert options["Melanoma, NOS"] == 4162276
    first = session.write(custom_definition(["Melanoma, NOS"], None, None, False, []))
    assert session.write(custom_definition(["Melanoma, NOS"], None, None, False, [])) == first
    second = session.write(custom_definition(["Melanoma, NOS"], 40, None, True, ["age_band"]))
    assert (first.name, second.name) == ("v1.yaml", "v2.yaml")
    session.approve(first)
    with pytest.raises(Refused):
        session.run(second)  # the approval was for v1's exact hash


def test_a_session_may_make_only_so_many_definitions(session: DemoSession) -> None:
    """A visitor cannot grow the work per Run without bound (security audit, item 47)."""
    from cdm.demo import MAX_DEFINITIONS

    paths = [
        session.write(custom_definition(["Melanoma, NOS"], age, None, False, []))
        for age in range(1, MAX_DEFINITIONS + 1)
    ]
    assert len(set(paths)) == MAX_DEFINITIONS
    # One already written is still found; a new one is refused with a clear message.
    assert session.write(custom_definition(["Melanoma, NOS"], 1, None, False, [])) == paths[0]
    with pytest.raises(Refused, match=f"already has {MAX_DEFINITIONS} cohort definitions"):
        session.write(custom_definition(["Melanoma, NOS"], MAX_DEFINITIONS + 1, None, False, []))
    assert len(list(session.folder.glob("*/v*.yaml"))) == MAX_DEFINITIONS


def test_the_scratch_folder_goes_with_the_session(tmp_path: Path) -> None:
    """A session's own temporary folder is deleted when the session is dropped or closed;
    a folder passed in is left alone."""
    import gc

    tables = fetch_tables(lambda collection: records())
    dropped = DemoSession(tables)
    folder = dropped.folder
    dropped.approve(next(p for p in library() if p.parent.name == "melanoma_any"))
    assert (folder / "approvals.jsonl").exists()
    del dropped
    gc.collect()
    assert not folder.exists()

    closed = DemoSession(tables)
    closed.close()
    assert not closed.folder.exists()

    given = DemoSession(tables, folder=tmp_path)
    given.approve(next(p for p in library() if p.parent.name == "melanoma_any"))
    given.close()
    assert (tmp_path / "approvals.jsonl").exists()


def test_the_public_page_hides_error_details() -> None:
    """Community Cloud runs from the repository root and reads .streamlit/config.toml there;
    with "none" the browser gets no traceback, exception type or message (item 47)."""
    import tomllib

    from cdm.demo import REPO

    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["showErrorDetails"] == "none"
