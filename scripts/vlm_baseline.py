"""Research Phase B on CPU, from the GPU job's BiomedCLIP embeddings (databricks/vlm_job.json).

Two steps, in order:

1. ``select``: train and validation only. Chooses the zero-shot prompt templates, the linear
   probe's hyperparameters and the Mahalanobis PCA size, and writes
   ``results/research/b_vlm/spec.json``. Commit it before step 2.
2. ``test``: refuses unless the spec is committed unchanged and no test look has happened.
   Scores the clients' test split once (plus the external sites and the drift sets), all in
   memory, then writes ``results.json`` and only after it the lock ``test_look.json``. A
   crash before the results are saved leaves nothing behind and the look unused; results
   without a lock still count as a look. EfficientNet's numbers come from the committed
   analysis-look predictions (no new look at any image), with intervals over the same
   resampling units.

``dry-run`` does step 2's whole computation on validation (and the drift selection draws)
instead, scoring no test or external-site image, and reports its time and peak memory.
The look itself runs as a Databricks serverless CPU job (``databricks/vlm_look_job.json``,
notebook ``vlm_look``) with this file: run ``preflight --code-version <commit>`` here first.

The embeddings stay outside git. Copy them once with
``databricks fs cp -r dbfs:/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings <dir>``
and pass ``--emb <dir>`` (default: the data folder's ``b_vlm``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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
# What the look runs: a change to any of these after the job's commit is refused.
LOOK_CODE = (
    "src",
    "scripts",
    "config",
    "databricks/notebooks/vlm_look.py",
    "results/research/b_vlm/spec.json",
)
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


@dataclass(frozen=True)
class Plan:
    """The embedding sets one run scores: the test look, or the dry run on validation."""

    client: str
    sites: tuple[str, ...]
    shifts: tuple[str, ...]
    # The committed EfficientNet predictions cover only the test look's images.
    effnet: bool


LOOK = Plan("test", ("buenos_aires", "pad_ufes"), SHIFTS, effnet=True)
# Validation, and the PathMNIST and CIFAR-10 selection draws: no test or external-site image.
DRY_RUN = Plan("val", (), ("pathmnist_select", "cifar10_select"), effnet=False)


def evaluation_rows(emb: Embeddings, plan: Plan = LOOK) -> dict[str, tuple[str, np.ndarray]]:
    """Each reported row: the embedding set it comes from and the mask of its images."""
    overlap = set(pd.read_csv(OVERLAP, dtype=str)["patient_id"])
    c = plan.client
    client_rows = emb.rows(c)
    rows = {
        f"client_{c}": (c, np.ones(len(client_rows), bool)),
        f"client_{c}_without_overlap_patients": (
            c,
            ~client_rows["patient_id"].isin(overlap).to_numpy(),
        ),
        f"ham10000_{c}": (c, (client_rows["source"] == "ham10000").to_numpy()),
    }
    for site in plan.sites:
        rows[site] = (site, emb.rows(site)["label"].notna().to_numpy())
    return rows


def check_inputs(emb: Embeddings, spec: dict[str, Any], plan: Plan) -> None:
    """The checks that need no git, so the Databricks job runs them too."""
    for key in ("job_run_id", "sha256"):
        if spec["embeddings"][key] != emb.summary[key]:
            sys.exit(f"these embeddings are not the ones the spec was chosen on ({key})")
    pca = pca_size_for_variance(emb.feats["train"].astype(np.float64))
    if pca != spec["drift"]["pca_components_method_d"]:
        sys.exit("the PCA size recomputed on train differs from the spec's")
    for name, (source, mask) in evaluation_rows(emb, plan).items():
        labels = emb.labels(source)[mask]
        if (labels < 0).any() or len(labels) == 0:
            sys.exit(f"{name}: unlabelled or empty")
        if plan.effnet:
            effnet_logits(emb.rows(source)[mask])


def preflight(emb: Embeddings, code_version: str | None = None) -> dict[str, Any]:
    """Everything that can fail without looking at a test result, run before any test image
    is scored. With ``code_version`` (the commit the Databricks job runs), the code and the
    spec must also be unchanged between that commit and HEAD."""
    if git("ls-files", "--", str(SPEC)) == "":
        sys.exit(f"{SPEC} must be committed before the test look")
    if git("status", "--porcelain", "--", "src", "scripts", "config", str(SPEC)):
        sys.exit("uncommitted changes in src/, scripts/, config/ or the spec: commit first")
    if code_version and git("diff", "--name-only", code_version, "HEAD", "--", *LOOK_CODE):
        sys.exit(f"src/, scripts/, config/ or the spec changed since {code_version}")
    for path in (LOCK, RESULTS):
        if path.exists() or git("log", "--oneline", "--", str(path)):
            sys.exit(f"the test look already happened (see {path} and its history)")
    spec: dict[str, Any] = json.loads(SPEC.read_text(encoding="utf-8"))
    check_inputs(emb, spec, LOOK)
    return spec


def compute(emb: Embeddings, spec: dict[str, Any], plan: Plan) -> dict[str, Any]:
    """Every metric of one run, in memory. It prints nothing and writes nothing, so a crash
    here leaves no result anywhere and the look unused. The bootstrap streams its resamples
    (one at a time, keeping only each resample's metric values), so memory does not grow
    with the number of resamples."""
    sets = [plan.client, *plan.sites, *plan.shifts]
    text = class_text(emb.text, template_choices(len(TEMPLATES))[spec["zero_shot"]["chosen"]])
    logits = {"zero_shot": {s: zero_shot_logits(emb.feats[s], text, emb.scale) for s in sets}}
    cfg = ProbeConfig(**spec["probe"]["chosen"])
    y_train = emb.labels("train")
    seeds = [f"probe_seed{s}" for s in spec["probe"]["seeds"]]
    for seed, name in zip(spec["probe"]["seeds"], seeds, strict=True):
        probe = train_probe(emb.feats["train"], y_train, len(CLASSES), seed, cfg)
        logits[name] = {s: probe.logits(emb.feats[s]) for s in sets}

    # The dry run has no EfficientNet predictions; the zero-shot logits stand in for them, so
    # it does the same amount of work as the look.
    reference = EFFNET_MODELS[0] if plan.effnet else f"stand_in_{EFFNET_MODELS[0]}"
    table: dict[str, dict[str, Any]] = {}
    for row, (source, mask) in evaluation_rows(emb, plan).items():
        sub = emb.rows(source)[mask]
        units = clusters(list(sub["patient_id"]), list(sub["lesion_id"]), list(sub["isic_id"]))
        labels = emb.labels(source)[mask]
        models = {m: v[source][mask] for m, v in logits.items()}
        if plan.effnet:
            models |= effnet_logits(sub)
        else:
            models |= {f"stand_in_{m}": models["zero_shot"] for m in EFFNET_MODELS}
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
        entry["paired_vs_" + reference] = {
            m: paired_bootstrap(models[m], models[reference], labels, units, n=BOOTSTRAP)
            for m in ("zero_shot", *seeds)
        }
        table[row] = entry

    x_train = emb.feats["train"].astype(np.float64)
    feats = {s: emb.feats[s].astype(np.float64) for s in [plan.client, *plan.shifts]}
    by_logits = {
        m: drift_metrics(x_train, y_train, feats, v, plan.shifts, in_distribution=plan.client)
        for m, v in logits.items()
    }
    first = by_logits["zero_shot"]
    drift: dict[str, Any] = {
        "in_distribution": plan.client,
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
                for shift in plan.shifts
            }
            for d in ("max_softmax", "energy")
        },
    }
    return {"classification": table, "drift": drift}


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Written in full to a temporary file, flushed to disk, then renamed into place, so the
    file is either complete or absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    json.loads(path.read_text(encoding="utf-8"))


def record(body: dict[str, Any], look: dict[str, Any], out: Path) -> None:
    """The results first, then the lock: the lock exists only once the results are saved."""
    results = {
        "spec": SPEC.relative_to(REPO).as_posix(),
        "spec_sha256": look["spec_sha256"],
        "code_version": look["code_version"],
        "embeddings_job_run_id": look["embeddings_job_run_id"],
        "ran_on": look["ran_on"],
        **body,
        "caveat": CAVEAT,
        "note": "Research use only, per the BiomedCLIP model card; never served.",
    }
    write_json(out / RESULTS.name, results)
    write_json(
        out / LOCK.name,
        {
            **look,
            "results_sha256": sha256(out / RESULTS.name),
            "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def new_look(emb: Embeddings, code_version: str, ran_on: str) -> dict[str, Any]:
    return {
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec_sha256": text_sha256(SPEC),
        "code_version": code_version,
        "embeddings_job_run_id": emb.summary["job_run_id"],
        "ran_on": ran_on,
    }


def test(emb: Embeddings) -> None:
    spec = preflight(emb)
    look = new_look(emb, git("rev-parse", "--short", "HEAD"), "this computer (CPU)")
    record(compute(emb, spec, LOOK), look, OUT)
    print(f"wrote {RESULTS}, then {LOCK}")


def peak_memory_mb() -> float:
    """This process's peak resident memory so far."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                *[(f"other{i}", ctypes.c_size_t) for i in range(7)],
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not info(current(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return float(c.peak_working_set) / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def dry_run(emb: Embeddings, out: Path) -> dict[str, Any]:
    """The look's whole computation on validation instead of test, to measure time and peak
    memory. It scores no test or external-site image and writes no lock."""
    if out.resolve().is_relative_to(OUT.resolve()):
        sys.exit(f"the dry run writes outside {OUT}")
    spec: dict[str, Any] = json.loads(SPEC.read_text(encoding="utf-8"))
    check_inputs(emb, spec, DRY_RUN)
    start = time.time()
    body = compute(emb, spec, DRY_RUN)
    usage = {"seconds": round(time.time() - start, 1), "peak_memory_mb": round(peak_memory_mb())}
    write_json(out / "dry_run.json", {"dry_run": True, **usage, **body})
    print(json.dumps(usage))
    return usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["select", "dry-run", "preflight", "test"])
    parser.add_argument("--emb", type=Path, default=data_root() / "b_vlm")
    parser.add_argument("--out", type=Path, default=data_root() / "b_vlm_dry_run")
    parser.add_argument("--code-version", help="preflight: the commit the Databricks job runs")
    args = parser.parse_args()
    emb = Embeddings(args.emb)
    if args.step == "select":
        select(emb)
    elif args.step == "dry-run":
        dry_run(emb, args.out)
    elif args.step == "preflight":
        preflight(emb, args.code_version)
        print("preflight passed: no test image has been scored")
    else:
        test(emb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
