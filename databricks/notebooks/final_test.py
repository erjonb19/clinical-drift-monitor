# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: the single look at test (selected configuration, 3 seeds)
# MAGIC Scores the four clients' pooled test split once, for each seed's checkpoint (refusing a
# MAGIC weight-hash mismatch) and for the 3-seed ensemble (mean of class probabilities):
# MAGIC per-client and Barcelona bordered/non-bordered scores, mean and range over seeds,
# MAGIC bootstrap 95% intervals by lesion, and calibration (ECE, NLL) before and after a
# MAGIC temperature fitted on validation. Lesion-level averaging is reported as a separately
# MAGIC labelled supplementary row: it was not selected by the tuning ladder. Federated
# MAGIC models (FedAvg, FedProx), if listed, are scored by the same code in the same look.
# MAGIC
# MAGIC Models come from the manifests the overnight GPU task writes; the run stops unless all
# MAGIC three control seeds are listed.
# MAGIC
# MAGIC Test is locked per model: each control seed, the ensemble, and each FedAvg and FedProx
# MAGIC run gets exactly one look. A lock file per model is written just before its first
# MAGIC test image is scored; a later run scores only models without a lock (for example a
# MAGIC federated run that finished late), unless `allow_second_look=true`, which is recorded.
# MAGIC The ensemble is scored only while its own lock is absent; it reuses the seeds' test
# MAGIC outputs, and per-seed results are reported only for seeds scored in this run.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/final_models.json",
    "federated_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/federated_models.json",
    "bootstrap": "1000",
    "allow_second_look": "false",
    # "" is v1. "v2" prefixes every lock and reports on v1's exact test images (main) and on
    # the full v2 test set (second row).
    "version": "",
    "v1_snapshot": "workspace.cdm.silver_images_v1_snapshot",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")], check=True
)

# COMMAND ----------

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.eval import classification_metrics, summarize
from cdm.final import bootstrap_by_lesion, calibration, ensemble_logits
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.posthoc import lesion_average
from cdm.report import score_rows, support
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
MODELS_FILE, FED_FILE = (
    Path(dbutils.widgets.get("models_file")),
    Path(dbutils.widgets.get("federated_file")),
)
if not MODELS_FILE.exists():
    raise RuntimeError(f"no control-seed manifest at {MODELS_FILE}: the GPU task did not finish")
MODELS = json.loads(MODELS_FILE.read_text())
if sorted(m["seed"] for m in MODELS) != [0, 1, 2]:
    raise RuntimeError(f"need control seeds 0, 1 and 2; manifest has {[m['seed'] for m in MODELS]}")
FEDERATED = json.loads(FED_FILE.read_text()) if FED_FILE.exists() else []
N_BOOT = int(dbutils.widgets.get("bootstrap"))
SECOND_LOOK = dbutils.widgets.get("allow_second_look").lower() == "true"
LOCKS = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2/test_looks")
KEYS = ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity", "accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 0  # serverless CPU has too little shared memory for DataLoader workers


VERSION = dbutils.widgets.get("version")
PREFIX = f"{VERSION}-" if VERSION else ""


def key(m: dict[str, object]) -> str:
    return f"{PREFIX}{m['kind']}-seed{m['seed']}"


ENSEMBLE = f"{PREFIX}ensemble-final-seeds0-1-2"
locked = {f.stem for f in LOCKS.glob("*.json")} if LOCKS.exists() else set()
fresh = lambda k: SECOND_LOOK or k not in locked  # noqa: E731
score_seeds = [m for m in MODELS if fresh(key(m))]
score_ensemble = fresh(ENSEMBLE)
FEDERATED = [m for m in FEDERATED if fresh(key(m))]
if not score_seeds and not score_ensemble and not FEDERATED:
    raise RuntimeError(f"every listed model already had its test look: {sorted(locked)}")
result: dict[str, object] = {
    "code_version": dbutils.widgets.get("code_version"),
    "config": CONFIG,
    "models": MODELS,
    "federated_models": FEDERATED,
    "device": str(device),
    "locked_before_this_run": sorted(locked),
    "second_look_allowed": SECOND_LOOK,
    "scored_now": [key(m) for m in score_seeds]
    + ([ENSEMBLE] if score_ensemble else [])
    + [key(m) for m in FEDERATED],
}

# COMMAND ----------

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
result["support"] = support(rows)
index = {c: i for i, c in enumerate(CLASSES)}
frames = {s: rows[rows["split"] == s].reset_index(drop=True) for s in ("val", "test")}
lesions = frames["test"]["lesion_id"].fillna(frames["test"]["isic_id"]).tolist()
result["test_images"], result["test_lesions"] = len(frames["test"]), len(set(lesions))
V1_IDS: set[str] = set()
if VERSION:
    V1_IDS = {
        r[0]
        for r in spark.read.table(dbutils.widgets.get("v1_snapshot")).select("isic_id").collect()
    }


def logits_for(model: torch.nn.Module, split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = frames[split]
    data = EncodedImages(
        list(frame["image"]),
        frame["label"].map(index).tolist(),
        eval_transform(CONFIG.get("image_size", 224), CONFIG.get("color_constancy", False)),
    )
    _, logits, labels = extract(model, DataLoader(data, batch_size=64, num_workers=WORKERS), device)
    return logits, labels


# Validation logits first (temperature fitting needs them); the locks are written just
# before the first test image is scored.
def load(m: dict[str, object]) -> torch.nn.Module:
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"{m} checkpoint mismatch"
    return model.to(device).eval()


val_logits, test_logits = [], []
models = []
for m in MODELS if (score_seeds or score_ensemble) else []:  # skip fully locked controls
    models.append(load(m))
    val_logits.append(logits_for(models[-1], "val")[0])
val_labels = frames["val"]["label"].map(index).to_numpy()
fed_models, fed_val = [], []
for m in FEDERATED:
    fed_models.append(load(m))
    fed_val.append(logits_for(fed_models[-1], "val")[0])

LOCKS.mkdir(parents=True, exist_ok=True)
for k in result["scored_now"]:
    (LOCKS / f"{k}.json").write_text(json.dumps({
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": result["code_version"], "second_look": k in locked}))  # fmt: skip
test_labels = frames["test"]["label"].map(index).to_numpy()
if score_seeds or score_ensemble:  # locked seeds are re-run only to build a fresh ensemble
    for model in models:
        test_logits.append(logits_for(model, "test")[0])
fed_test = [logits_for(model, "test")[0] for model in fed_models]

# COMMAND ----------


def build(mask: np.ndarray) -> dict[str, object]:
    """Every test report, on the test rows selected by ``mask``."""
    t_frame = frames["test"][mask].reset_index(drop=True)
    t_labels = test_labels[mask]
    t_lesions = [lesions[i] for i in np.flatnonzero(mask)]
    out: dict[str, object] = {"test_images": int(mask.sum()), "test_lesions": len(set(t_lesions))}

    def report(val: np.ndarray, test: np.ndarray) -> dict[str, object]:
        test = test[mask]
        return {
            "val_pooled": {k: classification_metrics(val_labels, val)[k] for k in KEYS},
            "scores": score_rows(t_frame, test, t_labels)["test"],
            "bootstrap_95": bootstrap_by_lesion(test, t_labels, t_lesions, n=N_BOOT),
            "calibration": calibration(val, val_labels, test, t_labels),
        }

    now = {key(m) for m in score_seeds}
    per_seed = []
    for m, val, test in zip(MODELS, val_logits, test_logits, strict=False):
        if key(m) in now:
            per_seed.append({"seed": m["seed"], **report(val, test)})
    out["single_model_per_seed"] = per_seed

    def pooled(entry: dict[str, object], k: str) -> float:
        return entry["scores"]["pooled"][k]

    if per_seed:
        out["single_model_over_seeds"] = {
            "seeds": [e["seed"] for e in per_seed],
            **{k: summarize([pooled(e, k) for e in per_seed]) for k in KEYS},
            **{
                f"calibration_{k}": summarize([e["calibration"][k] for e in per_seed])
                for k in ("ece_before", "ece_after", "nll_before", "nll_after")
            },
        }
    if score_ensemble:
        out["ensemble_3_seeds"] = report(ensemble_logits(val_logits), ensemble_logits(test_logits))

    # Supplementary, not selected: lesion averaging failed the keep rule on validation.
    avg_pairs = [
        (m, lesion_average(t[mask], t_lesions))
        for m, t in zip(MODELS, test_logits, strict=False)
        if key(m) in now
    ]
    avg_seeds = [a for _, a in avg_pairs]
    supplementary: dict[str, object] = {
        "note": "Not the selected configuration: lesion averaging missed the keep rule on "
        "validation (+0.83 points balanced accuracy). Shown for reference only.",
        "single_model_per_seed": [
            {"seed": m["seed"], **{k: classification_metrics(t_labels, a)[k] for k in KEYS}}
            for m, a in avg_pairs
        ],
    }
    if avg_seeds:
        supplementary["single_model_over_seeds"] = {
            k: summarize([classification_metrics(t_labels, a)[k] for a in avg_seeds]) for k in KEYS
        }
    if score_ensemble:
        avg_ens = lesion_average(ensemble_logits(test_logits)[mask], t_lesions)
        supplementary["ensemble_3_seeds"] = {
            **{k: classification_metrics(t_labels, avg_ens)[k] for k in KEYS},
            "bootstrap_95": bootstrap_by_lesion(avg_ens, t_labels, t_lesions, n=N_BOOT),
        }
    out["supplementary_lesion_average_not_selected"] = supplementary

    # Federated models, same scoring code: per seed, then mean and range per method.
    federated: dict[str, object] = {}
    for method in ("fedavg", "fedprox"):
        entries = [
            {"seed": m["seed"], **report(v, t)}
            for m, v, t in zip(FEDERATED, fed_val, fed_test, strict=True)
            if m["kind"] == method
        ]
        if entries:
            federated[method] = {
                "seeds": [e["seed"] for e in entries],
                "per_seed": entries,
                "over_seeds": {k: summarize([pooled(e, k) for e in entries]) for k in KEYS},
            }
    out["federated"] = federated
    return out


everything = np.ones(len(frames["test"]), dtype=bool)
if VERSION:
    # Main comparison: v1's exact test images; the full v2 test set is a second row.
    v1_test = frames["test"]["isic_id"].isin(V1_IDS).to_numpy()
    result["views"] = {"test_v1_images": build(v1_test), "test_full_v2": build(everything)}
else:
    result.update(build(everything))

print("FINAL_TEST_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
