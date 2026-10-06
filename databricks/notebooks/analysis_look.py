# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 analysis look (CPU), spec part B: `results/phase2/analysis_look/spec.json`
# MAGIC One look per model (own lock: started, then completed once results read back) for the
# MAGIC shipped ensemble and the three v2 centralized models. Scores the clients' test split and
# MAGIC every Buenos Aires and PAD-UFES-20 image; saves per-image probabilities; reports
# MAGIC seven-class accuracy at the external sites over the classes present (mapping and
# MAGIC exclusions fixed in the spec), accuracy by skin type within each site, the 20 worst missed
# MAGIC melanomas on the client test split (IDs and metadata only), and a check that the reported
# MAGIC client-test numbers (also scored on CPU) reproduce. Descriptive only: nothing here changes a model or threshold.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "reported_test": "/Volumes/workspace/cdm/raw/results/phase2/analysis_look/reported_test.json",
    "bootstrap": "1000",
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
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.eval import classification_metrics, softmax
from cdm.final import bootstrap_by_lesion, ensemble_logits
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
CODE = dbutils.widgets.get("code_version")
N_BOOT = int(dbutils.widgets.get("bootstrap"))
MODELS = sorted(json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
                key=lambda m: m["seed"])  # fmt: skip
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/analysis_look")
LOCKS = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2/analysis_looks")
OUT.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)
ENS = "v2-ensemble-final-seeds0-1-2"
KEYS = [ENS, *(f"v2-final-seed{m['seed']}" for m in MODELS)]
INDEX, MEL = {c: i for i, c in enumerate(CLASSES)}, CLASSES.index("mel")
SITES = ("buenos_aires", "pad_ufes")
for k in KEYS:
    if (LOCKS / f"{k}.json").exists():
        raise RuntimeError(
            f"{k} already had its analysis look: {(LOCKS / f'{k}.json').read_text()}"
        )

images = spark.read.table("workspace.cdm.silver_images").select(
    "isic_id", "source", "role", "client", "split", "label", "lesion_id", "image"
)
meta = spark.read.table("workspace.cdm.silver_metadata").select(
    "isic_id", "source", "fitzpatrick_skin_type", "image_type", "anatom_site_1", "age_approx",
    "sex", "diagnosis_3")  # fmt: skip
rows = (
    images.join(meta, ["source", "isic_id"], "left")
    .where("(role = 'client' AND split = 'test' AND label IS NOT NULL) "
           "OR source IN ('buenos_aires', 'pad_ufes')")
    .orderBy("source", "isic_id")
    .toPandas()
)  # fmt: skip
rows["lesion"] = rows["lesion_id"].fillna(rows["isic_id"])
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
rows["skin_type"] = rows["fitzpatrick_skin_type"].fillna("not recorded")
labels = rows["label"].map(INDEX).fillna(-1).astype(int).to_numpy()
looks = {}
for k in KEYS:
    looks[k] = {"status": "started", "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "code_version": CODE}  # fmt: skip
    (LOCKS / f"{k}.json").write_text(json.dumps(looks[k]))

# COMMAND ----------

data = EncodedImages(list(rows["image"]), labels.tolist(),
                     eval_transform(CONFIG.get("image_size", 224)))  # fmt: skip
logits = {}
for m in MODELS:
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], m
    _, z, _ = extract(model.eval(), DataLoader(data, batch_size=64, num_workers=0),
                      torch.device("cpu"))  # fmt: skip
    logits[f"v2-final-seed{m['seed']}"] = z
logits[ENS] = ensemble_logits([logits[f"v2-final-seed{m['seed']}"] for m in MODELS])
probs = {k: softmax(v) for k, v in logits.items()}

pred = rows[["isic_id", "source", "client", "split", "label", "lesion", "skin_type",
             "image_type", "bordered"]].copy()  # fmt: skip
for k in KEYS:
    for i, c in enumerate(CLASSES):
        pred[f"{k}__{c}"] = probs[k][:, i]
local = Path(tempfile.gettempdir()) / "predictions.csv.gz"
pred.to_csv(local, index=False)
(OUT / "predictions.csv.gz").write_bytes(local.read_bytes())

# COMMAND ----------

KEYM = ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity", "accuracy")


def metrics(mask: np.ndarray, k: str, boot: bool = True) -> dict[str, object]:
    mask = mask & (labels >= 0)
    m = classification_metrics(labels[mask], logits[k][mask])
    out = {"images": int(mask.sum()), "melanoma_images": int((labels[mask] == MEL).sum()),
           "classes_present": [CLASSES[c] for c in sorted(set(labels[mask]))],
           **{x: m[x] for x in KEYM}, "recall": m["recall"]}  # fmt: skip
    if boot:
        out["bootstrap_95"] = bootstrap_by_lesion(logits[k][mask], labels[mask],
                                                  rows.loc[mask, "lesion"].tolist(), n=N_BOOT)  # fmt: skip
    return out


is_test = ((rows["role"] == "client") & (rows["split"] == "test")).to_numpy()
reported = json.loads(Path(dbutils.widgets.get("reported_test")).read_text())
repro = {}
for k in KEYS:
    m = metrics(is_test, k, boot=False)
    repro[k] = {x: {"cpu": m[x], "reported": reported[k][x],
                    "difference_points": 100 * (m[x] - reported[k][x])}
                for x in ("balanced_accuracy", "melanoma_sensitivity")}  # fmt: skip

external, skin = {}, {}
for site in SITES:
    at = (rows["source"] == site).to_numpy()
    external[site] = {k: metrics(at, k) for k in KEYS}
    lab = at & (labels >= 0)
    counts = rows.loc[lab, "skin_type"].value_counts().to_dict()
    groups = {}
    for t, n in sorted(counts.items()):
        if t in ("IV", "V", "VI") or t == "not recorded" or n < 100:
            continue
        g = metrics(at & (rows["skin_type"] == t).to_numpy(), ENS)
        g["too_small_to_interpret"] = g["melanoma_images"] < 20
        groups[t] = g
    skin[site] = {"labelled_images_by_type": {str(t): int(n) for t, n in counts.items()},
                  "counts_only_gap": {t: int(counts.get(t, 0)) for t in ("IV", "V", "VI")},
                  "ensemble": groups}  # fmt: skip

p_mel = probs[ENS][:, MEL]
missed = is_test & (labels == MEL) & (probs[ENS].argmax(1) != MEL)
order = np.argsort(p_mel[missed])[:20]
cols = ["isic_id", "client", "skin_type", "image_type", "anatom_site_1", "age_approx", "sex",
        "diagnosis_3", "bordered"]  # fmt: skip
worst = rows.loc[missed, cols].iloc[order].assign(
    predicted=[CLASSES[c] for c in probs[ENS][missed].argmax(1)[order]],
    melanoma_probability=p_mel[missed][order],
)  # fmt: skip
result = {
    "spec": "results/phase2/analysis_look/spec.json, part B", "code_version": CODE,
    "device": "cpu", "scored_images": int(len(rows)),
    "reproducibility_check_client_test": repro,
    "external_seven_class": external,
    "skin_type_within_site": skin,
    "worst_missed_melanomas": {"definition": "client-test melanomas the ensemble misses "
                               "(predicted class not mel), lowest melanoma probability first",
                               "missed_total": int(missed.sum()),
                               "top20": json.loads(worst.to_json(orient="records"))},
    "note": "descriptive only; no model or threshold change; skin type is mixed up with site and "
            "imaging type",
}  # fmt: skip
(OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
assert json.loads((OUT / "results.json").read_text())["scored_images"] == len(rows)
for k in KEYS:
    (LOCKS / f"{k}.json").write_text(json.dumps({**looks[k], "status": "completed",
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds")}))  # fmt: skip
dbutils.notebook.exit(json.dumps({"scored": len(rows), "reproducibility": repro}))
