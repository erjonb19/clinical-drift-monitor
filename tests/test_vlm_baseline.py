"""scripts/vlm_baseline.py: input checks and the refusals that protect the one test look."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from cdm.splits import CLASSES
from cdm.vlm import paired_bootstrap

REPO = Path(__file__).resolve().parents[1]


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vlm_baseline", REPO / "scripts" / "vlm_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def fake_embeddings(folder: Path, n: int = 4) -> None:
    folder.mkdir()
    meta = pd.DataFrame(
        {"set": "test", "isic_id": [f"I{i}" for i in range(n)], "label": "mel", "source": "x",
         "patient_id": None, "lesion_id": None}
    )  # fmt: skip
    meta.to_csv(folder / "meta.csv", index=False)
    np.savez(folder / "embeddings.npz", test=np.zeros((n, 3), np.float32))
    np.savez(folder / "text.npz", text=np.zeros((1, len(CLASSES), 3), np.float32))
    sha = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
           for f in ("meta.csv", "embeddings.npz", "text.npz")}  # fmt: skip
    summary = {"sha256": sha, "logit_scale": 10.0, "job_run_id": "1"}
    (folder / "summary.json").write_text(json.dumps(summary))


def test_embeddings_load_when_hashes_and_counts_match(tmp_path: Path) -> None:
    fake_embeddings(tmp_path / "e")
    emb = script().Embeddings(tmp_path / "e")
    assert emb.labels("test").tolist() == [CLASSES.index("mel")] * 4


def test_embeddings_refuse_a_file_that_changed(tmp_path: Path) -> None:
    fake_embeddings(tmp_path / "e")
    np.savez(tmp_path / "e" / "embeddings.npz", test=np.ones((4, 3), np.float32))
    with pytest.raises(SystemExit, match="does not match"):
        script().Embeddings(tmp_path / "e")


def predictions(path: Path, ids: list[str], labels: list[str]) -> Path:
    frame = pd.DataFrame({"isic_id": ids, "label": labels})
    for m in script().EFFNET_MODELS:
        for i, c in enumerate(CLASSES):
            frame[f"{m}__{c}"] = [0.9 if i == k else 0.1 / 6 for k in range(len(ids))]
    frame.to_csv(path, index=False)
    return path


def test_effnet_rows_follow_the_requested_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = script()
    monkeypatch.setattr(
        module, "EFFNET", predictions(tmp_path / "p.csv", ["a", "b", "c"], ["x"] * 3)
    )
    rows = pd.DataFrame({"isic_id": ["c", "a"], "label": ["x", "x"]})
    got = module.effnet_logits(rows)[module.EFFNET_MODELS[0]]
    assert got.argmax(1).tolist() == [2, 0]


def test_effnet_refuses_a_label_mismatch_or_a_missing_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = script()
    monkeypatch.setattr(
        module, "EFFNET", predictions(tmp_path / "p.csv", ["a", "b"], ["mel", "nv"])
    )
    with pytest.raises(ValueError, match="labels differ"):
        module.effnet_logits(pd.DataFrame({"isic_id": ["a"], "label": ["nv"]}))
    with pytest.raises(ValueError, match="no EfficientNet prediction"):
        module.effnet_logits(pd.DataFrame({"isic_id": ["z"], "label": ["nv"]}))


@pytest.mark.parametrize(
    ("outputs", "existing", "message"),
    [
        ({"ls-files": ""}, None, "must be committed"),
        ({"status": " M scripts/x.py"}, None, "uncommitted changes"),
        ({"diff": "src/cdm/vlm.py"}, None, "changed since"),
        ({}, "test_look.json", "already happened"),
        ({}, "results.json", "already happened"),
        ({"log": "abc1234 an earlier look"}, None, "already happened"),
    ],
)
def test_preflight_refuses_before_any_test_image_is_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outputs: dict[str, str],
    existing: str | None,
    message: str,
) -> None:
    module = script()
    defaults = {"ls-files": "spec.json", "status": "", "log": "", "diff": ""}
    monkeypatch.setattr(module, "git", lambda *a: {**defaults, **outputs}.get(a[0], ""))
    monkeypatch.setattr(module, "SPEC", tmp_path / "spec.json")
    monkeypatch.setattr(module, "LOCK", tmp_path / "test_look.json")
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(module, "check_inputs", lambda *a: pytest.fail("inputs checked"))
    (tmp_path / "spec.json").write_text("{}")
    if existing:
        (tmp_path / existing).write_text("{}")
    with pytest.raises(SystemExit, match=message):
        module.preflight(object(), code_version="abc1234")
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        ["spec.json", *([existing] if existing else [])]
    )


def look_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = script()
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "SPEC", tmp_path / "spec.json")
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(module, "LOCK", tmp_path / "out" / "test_look.json")
    monkeypatch.setattr(module, "RESULTS", tmp_path / "out" / "results.json")
    monkeypatch.setattr(module, "git", lambda *a: "abc1234")
    monkeypatch.setattr(module, "preflight", lambda emb: {})
    (tmp_path / "spec.json").write_text("{}")
    return module


class FakeEmb:
    summary = {"job_run_id": "1"}


def test_a_crash_while_scoring_leaves_no_lock_and_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = look_module(tmp_path, monkeypatch)

    def crash(*a: object) -> None:
        raise MemoryError

    monkeypatch.setattr(module, "compute", crash)
    with pytest.raises(MemoryError):
        module.test(FakeEmb())
    assert not (tmp_path / "out").exists()


def test_the_lock_is_written_after_the_results_and_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = look_module(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "compute", lambda *a: {"classification": {}, "drift": {}})
    order: list[str] = []
    real = module.write_json

    def logged(path: Path, value: dict[str, object]) -> None:
        order.append(path.name)
        real(path, value)

    monkeypatch.setattr(module, "write_json", logged)
    module.test(FakeEmb())
    assert order == ["results.json", "test_look.json"]
    lock = json.loads((tmp_path / "out" / "test_look.json").read_text())
    results = (tmp_path / "out" / "results.json").read_bytes()
    assert lock["results_sha256"] == hashlib.sha256(results).hexdigest()
    assert sorted(f.name for f in (tmp_path / "out").iterdir()) == [
        "results.json",
        "test_look.json",
    ]


def test_the_dry_run_never_writes_into_the_results_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = look_module(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="writes outside"):
        module.dry_run(FakeEmb(), tmp_path / "out" / "dry")


def test_peak_memory_is_measured() -> None:
    assert script().peak_memory_mb() > 1


def test_paired_bootstrap_differences_have_the_right_sign() -> None:
    labels = np.repeat(np.arange(len(CLASSES)), 20)
    right = np.eye(len(CLASSES))[labels] * 5
    wrong = np.eye(len(CLASSES))[(labels + 1) % len(CLASSES)] * 5
    out = paired_bootstrap(right, wrong, labels, [str(i) for i in range(len(labels))], n=50)
    assert out["balanced_accuracy"]["difference"] == 1.0
    assert out["balanced_accuracy"]["low"] == 1.0
    assert out["melanoma_sensitivity"]["high"] == 1.0
