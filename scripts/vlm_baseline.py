"""Research Phase B on CPU, from the GPU job's BiomedCLIP embeddings (databricks/vlm_job.json).

Two steps, in order:

1. ``select``: train and validation only. Chooses the zero-shot prompt templates, the linear
   probe's hyperparameters and the Mahalanobis PCA size, and writes
   ``results/research/b_vlm/spec.json``. Commit it before step 2.
2. ``test``: refuses unless the spec is committed unchanged and no test look has happened.
   Writes the look's lock first, then scores the clients' test split once (plus the
   external sites and the drift sets) and writes ``results.json``. EfficientNet's numbers
   come from the committed analysis-look predictions (no new look at any image), with
   intervals over the same resampling units.

The embeddings stay outside git. Copy them once with
``databricks fs cp -r dbfs:/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings <dir>``
and pass ``--emb <dir>`` (default: the data folder's ``b_vlm``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdm.data import data_root
from cdm.eval import classification_metrics, summarize
from cdm.final import bootstrap_by_lesion
from cdm.ood import pca_size_for_variance
from cdm.probe import SEEDS, ProbeConfig, select_config, train_probe
from cdm.splits import CLASSES
from cdm.vlm import (
    CLASS_TEXT,
    MODEL_VERSION,
    PINS,
    TEMPLATES,
    class_text,
    clusters,
    drift_metrics,
    paired_bootstrap,
    select_templates,
    template_choices,
    zero_shot_logits,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "research" / "b_vlm"
SPEC, LOCK, RESULTS = OUT / "spec.json", OUT / "test_look.json", OUT / "results.json"
EFFNET = REPO / "results" / "phase2" / "analysis_look" / "predictions.csv.gz"
OVERLAP = REPO / "config" / "patient_overlap_v1.csv"
EFFNET_MODELS = (
    "v2-ensemble-final-seeds0-1-2",
    "v2-final-seed0",
    "v2-final-seed1",
    "v2-final-seed2",
)
SHIFTS = ("buenos_aires", "pad_ufes", "pathmnist_eval", "cifar10_eval")
BOOTSTRAP = 1000
INDEX = {c: i for i, c in enumerate(CLASSES)}
CAVEAT = (
    "BiomedCLIP was pretrained on PMC-15M figure-caption pairs, which may include figures of "
    "ISIC or HAM10000 images, so zero-shot results cannot be shown to be on unseen images."
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(path: Path) -> str:
    """Of a text file as git stores it (LF line endings), so a checkout on any system matches."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class Embeddings:
    """The GPU job's files, checked against the hashes in its summary."""

    def __init__(self, folder: Path) -> None:
        self.summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for name, digest in self.summary["sha256"].items():
            if sha256(folder / name) != digest:
                sys.exit(f"{folder / name} does not match the job's summary")
        with np.load(folder / "embeddings.npz") as z:
            self.feats = {k: z[k] for k in z.files}
        with np.load(folder / "text.npz") as z:
            self.text = z["text"]
        self.meta = pd.read_csv(folder / "meta.csv", dtype=str)
        for key, frame in self.meta.groupby("set"):
            name = str(key)
            if len(frame) != len(self.feats[name]):
                sys.exit(f"{name}: {len(frame)} metadata rows, {len(self.feats[name])} embeddings")
        self.scale = float(self.summary["logit_scale"])

    def rows(self, name: str) -> pd.DataFrame:
        return self.meta[self.meta["set"] == name].reset_index(drop=True)

    def labels(self, name: str) -> np.ndarray:
        return self.rows(name)["label"].map(INDEX).fillna(-1).astype(np.int64).to_numpy()


def select(emb: Embeddings) -> None:
    x_train, y_train = emb.feats["train"], emb.labels("train")
    x_val, y_val = emb.feats["val"], emb.labels("val")
    assert (y_train >= 0).all() and (y_val >= 0).all(), "unlabelled client rows"
    choice, template_scores = select_templates(x_val, y_val, emb.text, emb.scale)
    cfg, grid = select_config(x_train, y_train, x_val, y_val, len(CLASSES))
    spec = {
        "committed_before_running": True,
        "decided": datetime.now(UTC).date().isoformat(),
        "code_version": git("rev-parse", "--short", "HEAD"),
        "model": {"version": MODEL_VERSION, "pins": PINS, "logit_scale": emb.scale},
        "embeddings": {
            k: emb.summary[k] for k in ("job_run_id", "code_version", "images", "sha256")
        },
        "zero_shot": {
            "class_text": CLASS_TEXT,
            "templates": list(TEMPLATES),
            "chosen": choice,
            "chosen_templates": [TEMPLATES[i] for i in template_choices(len(TEMPLATES))[choice]],
            "validation_balanced_accuracy": template_scores,
        },
        "probe": {
            "chosen": {k: getattr(cfg, k) for k in ("lr", "weight_decay", "epochs", "batch_size")},
            "seeds": list(SEEDS),
            "validation_grid": grid,
            "rule": "highest mean validation balanced accuracy over the seeds; class-weighted "
            "cross-entropy; features standardized on train",
        },
        "drift": {
            "detectors": ["mahalanobis (PCA by method D, fit on train)", "max_softmax", "energy"],
            "pca_components_method_d": pca_size_for_variance(x_train.astype(np.float64)),
            "logits": "zero-shot, and each probe seed",
            "in_distribution": "the clients' test split",
            "shifts": list(SHIFTS),
            "gram": "not applicable: it needs convolutional feature maps and BiomedCLIP is a ViT",
            "not_used": "the PathMNIST and CIFAR-10 selection draws were embedded but are not "
            "used: Phase 2's method C (PCA chosen on them) is not run for BiomedCLIP",
            "mahalanobis": "reported once: it uses features only, not logits",
        },
        "test_look": {
            "lock": LOCK.relative_to(REPO).as_posix(),
            "rows": {
                "client_test": "the clients' pooled test split",
                "client_test_without_overlap_patients": "minus images of the patients in "
                "config/patient_overlap_v1.csv",
                "ham10000_test": "the HAM10000 part of the test split",
                "buenos_aires": "labelled images, the analysis look's seven-class mapping",
                "pad_ufes": "labelled images, the analysis look's seven-class mapping",
            },
            "metrics": "balanced accuracy (over the classes present), melanoma sensitivity, "
            "macro AUROC, per-class recall",
            "bootstrap": f"{BOOTSTRAP} resamples, seed 0, 95% percentile intervals; resampling "
            "unit: the patient where ISIC records one, else the lesion, else the image",
            "comparison": f"EfficientNet {', '.join(EFFNET_MODELS)} from "
            f"{EFFNET.relative_to(REPO).as_posix()} (sha256 {sha256(EFFNET)}), no new look",
            "seeds": "probe results as each seed, and mean and range over the 3 seeds",
            "paired": f"differences against {EFFNET_MODELS[0]} in balanced accuracy and "
            "melanoma sensitivity, both scored on the same resamples (paired bootstrap)",
            "lock_rule": "the lock file is committed with the results; the look is refused "
            "if the lock exists or has git history, if src/, scripts/, config/ or the spec "
            "have uncommitted changes, or if the embeddings are not the spec's",
        },
        "caveat": CAVEAT,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {SPEC}: zero-shot {choice}, probe {cfg}")
    print("commit it, then run: python scripts/vlm_baseline.py test")


def effnet_logits(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Log-probabilities of the committed EfficientNet predictions, row for row with ``rows``
    (matched by ISIC ID, and the label cross-checked against silver's)."""
    pred = pd.read_csv(EFFNET, dtype={"label": str}).set_index("isic_id")
    missing = set(rows["isic_id"]) - set(pred.index)
    if missing:
        raise ValueError(f"{len(missing)} images have no EfficientNet prediction")
    pred = pred.loc[list(rows["isic_id"])]
    ours, theirs = rows["label"].fillna("").to_numpy(), pred["label"].fillna("").to_numpy()
    if not np.array_equal(ours, theirs):
        raise ValueError(f"{int((ours != theirs).sum())} labels differ from the predictions file")
    return {
        m: np.log(np.clip(pred[[f"{m}__{c}" for c in CLASSES]].to_numpy(np.float64), 1e-12, 1))
        for m in EFFNET_MODELS
    }


def scored(logits: np.ndarray, labels: np.ndarray, units: list[str]) -> dict[str, Any]:
    point = classification_metrics(labels, logits)
    return {**point, "bootstrap_95": bootstrap_by_lesion(logits, labels, units, n=BOOTSTRAP)}


def evaluation_rows(emb: Embeddings) -> dict[str, tuple[str, np.ndarray]]:
    """Each reported row: the embedding set it comes from and the mask of its images."""
    overlap = set(pd.read_csv(OVERLAP, dtype=str)["patient_id"])
    test_rows = emb.rows("test")
    rows = {
        "client_test": ("test", np.ones(len(test_rows), bool)),
        "client_test_without_overlap_patients": (
            "test",
            ~test_rows["patient_id"].isin(overlap).to_numpy(),
        ),
        "ham10000_test": ("test", (test_rows["source"] == "ham10000").to_numpy()),
    }
    for site in ("buenos_aires", "pad_ufes"):
        rows[site] = (site, emb.rows(site)["label"].notna().to_numpy())
    return rows


def preflight(emb: Embeddings) -> dict[str, Any]:
    """Everything that can fail without looking at a test result, run before the lock is
    written, so a broken input cannot use up the one look."""
    if git("ls-files", "--", str(SPEC)) == "":
        sys.exit(f"{SPEC} must be committed before the test look")
    if git("status", "--porcelain", "--", "src", "scripts", "config", str(SPEC)):
        sys.exit("uncommitted changes in src/, scripts/, config/ or the spec: commit first")
    if LOCK.exists() or git("log", "--oneline", "--", str(LOCK)):
        sys.exit(f"the test look already happened (see {LOCK} and its history)")
    spec: dict[str, Any] = json.loads(SPEC.read_text(encoding="utf-8"))
    for key in ("job_run_id", "sha256"):
        if spec["embeddings"][key] != emb.summary[key]:
            sys.exit(f"these embeddings are not the ones the spec was chosen on ({key})")
    pca = pca_size_for_variance(emb.feats["train"].astype(np.float64))
    if pca != spec["drift"]["pca_components_method_d"]:
        sys.exit("the PCA size recomputed on train differs from the spec's")
    for name, (source, mask) in evaluation_rows(emb).items():
        labels = emb.labels(source)[mask]
        if (labels < 0).any() or len(labels) == 0:
            sys.exit(f"{name}: unlabelled or empty")
        effnet_logits(emb.rows(source)[mask])
    return spec


def test(emb: Embeddings) -> None:
    spec = preflight(emb)
    look = {
        "status": "started",
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec_sha256": text_sha256(SPEC),
        "code_version": git("rev-parse", "--short", "HEAD"),
        "embeddings_job_run_id": emb.summary["job_run_id"],
    }
    LOCK.write_text(json.dumps(look, indent=2) + "\n", encoding="utf-8", newline="\n")

    sets = ["train", "val", "test", *SHIFTS]
    text = class_text(emb.text, template_choices(len(TEMPLATES))[spec["zero_shot"]["chosen"]])
    logits = {"zero_shot": {s: zero_shot_logits(emb.feats[s], text, emb.scale) for s in sets}}
    cfg = ProbeConfig(**spec["probe"]["chosen"])
    y_train = emb.labels("train")
    seeds = [f"probe_seed{s}" for s in spec["probe"]["seeds"]]
    for seed, name in zip(spec["probe"]["seeds"], seeds, strict=True):
        probe = train_probe(emb.feats["train"], y_train, len(CLASSES), seed, cfg)
        logits[name] = {s: probe.logits(emb.feats[s]) for s in sets}

    table: dict[str, dict[str, Any]] = {}
    for row, (source, mask) in evaluation_rows(emb).items():
        sub = emb.rows(source)[mask]
        units = clusters(list(sub["patient_id"]), list(sub["lesion_id"]), list(sub["isic_id"]))
        labels = emb.labels(source)[mask]
        effnet = effnet_logits(sub)
        models = {m: v[source][mask] for m, v in logits.items()} | effnet
        entry: dict[str, Any] = {
            "images": int(mask.sum()),
            "units": len(set(units)),
            "models": {m: scored(z, labels, units) for m, z in models.items()},
        }
        entry["probe_over_seeds"] = {
            k: summarize([float(entry["models"][s][k]) for s in seeds])
            for k in ("balanced_accuracy", "melanoma_sensitivity")
            if all(entry["models"][s][k] is not None for s in seeds)
        }
        entry["paired_vs_" + EFFNET_MODELS[0]] = {
            m: paired_bootstrap(models[m], effnet[EFFNET_MODELS[0]], labels, units, n=BOOTSTRAP)
            for m in ("zero_shot", *seeds)
        }
        table[row] = entry

    x_train = emb.feats["train"].astype(np.float64)
    feats = {s: emb.feats[s].astype(np.float64) for s in ["test", *SHIFTS]}
    by_logits = {m: drift_metrics(x_train, y_train, feats, v, SHIFTS) for m, v in logits.items()}
    first = by_logits["zero_shot"]
    drift: dict[str, Any] = {
        "pca_components_method_d": first["pca_components_method_d"],
        "mahalanobis": first["metrics"]["mahalanobis"],
        "zero_shot": {d: first["metrics"][d] for d in ("max_softmax", "energy")},
        "probe_by_seed": {
            s: {d: by_logits[s]["metrics"][d] for d in ("max_softmax", "energy")} for s in seeds
        },
        "probe_over_seeds": {
            d: {
                shift: {
                    k: summarize([by_logits[s]["metrics"][d][shift][k] for s in seeds])
                    for k in ("auroc", "fpr95")
                }
                for shift in SHIFTS
            }
            for d in ("max_softmax", "energy")
        },
    }

    results = {
        "spec": SPEC.relative_to(REPO).as_posix(),
        "spec_sha256": look["spec_sha256"],
        "code_version": look["code_version"],
        "embeddings_job_run_id": emb.summary["job_run_id"],
        "device": "cpu (probe and scoring); embeddings from the GPU job",
        "classification": table,
        "drift": drift,
        "caveat": CAVEAT,
        "note": "Research use only, per the BiomedCLIP model card; never served.",
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    json.loads(RESULTS.read_text(encoding="utf-8"))
    look.update(status="completed", completed_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    LOCK.write_text(json.dumps(look, indent=2) + "\n", encoding="utf-8", newline="\n")
    head = table["client_test"]["models"]
    for m in ("zero_shot", seeds[0], EFFNET_MODELS[0]):
        print(
            m, round(head[m]["balanced_accuracy"], 4), head[m]["bootstrap_95"]["balanced_accuracy"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["select", "test"])
    parser.add_argument("--emb", type=Path, default=data_root() / "b_vlm")
    args = parser.parse_args()
    emb = Embeddings(args.emb)
    select(emb) if args.step == "select" else test(emb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
