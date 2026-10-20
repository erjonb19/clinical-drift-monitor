"""Results files logged to MLflow: metric names, which values count, and the hash check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdm.evidence import INDEX, metric_key, numeric_leaves, sha256, stale_entries

REPO = Path(__file__).resolve().parents[1]


def test_numeric_leaves_keeps_finite_numbers_by_path_only() -> None:
    data = {
        "a": {"auroc": 0.75, "n": 12, "ok": True, "name": "x", "none": None},
        "rows": [{"v": 1.5}, {"v": float("nan")}],
        "bad key (%)": 2,
    }
    assert numeric_leaves(data) == {
        "a/auroc": 0.75,
        "a/n": 12.0,
        "rows/0/v": 1.5,
        "bad key ___": 2.0,
    }


def test_disallowed_characters_are_replaced_and_long_names_refused() -> None:
    assert metric_key(["x", "a+b", "c@d"]) == "x/a_b/c_d"
    with pytest.raises(ValueError):
        metric_key(["k" * 251])


def test_hash_ignores_line_endings(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert sha256(lf) == sha256(crlf)


def test_stale_entries_catch_a_result_edited_after_logging(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text('{"auroc": 0.7}\n', encoding="utf-8")
    (tmp_path / INDEX).parent.mkdir(parents=True)
    entry = {"run_id": "x", "file_sha256": sha256(tmp_path / "r.json")}
    (tmp_path / INDEX).write_text(json.dumps({"r.json": entry}), encoding="utf-8")
    assert stale_entries(tmp_path) == []
    (tmp_path / "r.json").write_text('{"auroc": 0.9}\n', encoding="utf-8")
    assert stale_entries(tmp_path) == ["r.json"]


def test_every_committed_result_in_the_index_matches_what_was_logged() -> None:
    assert stale_entries(REPO) == []
