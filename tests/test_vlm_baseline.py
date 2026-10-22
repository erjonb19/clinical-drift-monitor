"""scripts/vlm_baseline.py: input checks and the refusals that protect the one test look."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    ("outputs", "lock_exists", "message"),
    [
        ({"ls-files": ""}, False, "must be committed"),
        ({"status": " M scripts/x.py"}, False, "uncommitted changes"),
        ({}, True, "already happened"),
        ({"log": "abc1234 an earlier look"}, False, "already happened"),
    ],
)
def test_preflight_refuses_before_any_lock_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outputs: dict[str, str],
    lock_exists: bool,
    message: str,
) -> None:
    module = script()
    defaults = {"ls-files": "spec.json", "status": "", "log": ""}
    monkeypatch.setattr(module, "git", lambda *a: {**defaults, **outputs}.get(a[0], ""))
    monkeypatch.setattr(module, "SPEC", tmp_path / "spec.json")
    monkeypatch.setattr(module, "LOCK", tmp_path / "test_look.json")
    (tmp_path / "spec.json").write_text("{}")
    if lock_exists:
        (tmp_path / "test_look.json").write_text("{}")
    with pytest.raises(SystemExit, match=message):
        module.preflight(object())
    assert lock_exists == (tmp_path / "test_look.json").exists()


def test_paired_bootstrap_differences_have_the_right_sign() -> None:
    labels = np.repeat(np.arange(len(CLASSES)), 20)
    right = np.eye(len(CLASSES))[labels] * 5
    wrong = np.eye(len(CLASSES))[(labels + 1) % len(CLASSES)] * 5
    out = paired_bootstrap(right, wrong, labels, [str(i) for i in range(len(labels))], n=50)
    assert out["balanced_accuracy"]["difference"] == 1.0
    assert out["balanced_accuracy"]["low"] == 1.0
    assert out["melanoma_sensitivity"]["high"] == 1.0
