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
