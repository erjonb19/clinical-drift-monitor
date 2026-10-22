# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase D: Grad-CAM review worksheet (serverless CPU)
# MAGIC Under `results/research/d_review/spec.json`. Picks the client-test images to review from
# MAGIC the committed predictions and the Phase 2 drift run's saved scores (the worst misses per
# MAGIC class and the most out-of-distribution), checks that the shipped seed-0 checkpoint
# MAGIC reproduces its committed probabilities on them, then draws Grad-CAM maps for the predicted
# MAGIC and true class with the randomization sanity check and agreement across seeds.
# MAGIC
# MAGIC Writes overlays to the volume, an HTML worksheet to MLflow (with license and attribution per
# MAGIC image), `workspace.cdm.gold_review_worksheet` with a blank `reviewer_notes` column, and the
# MAGIC IDs-and-values CSV and results for the repository. Nothing is "reviewed" until a person
# MAGIC fills in the notes. Grad-CAM on EfficientNet-B0's last block is 7x7 at 224 px: coarse.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "effnet_scores": "/Volumes/workspace/cdm/raw/results/phase2/drift/v2",
    "out": "/Volumes/workspace/cdm/raw/results/research/d_review",
    "review_root": "/Volumes/workspace/cdm/raw/review",
    "experiment_id": "3252981153928765",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import base64
import hashlib
import html
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyspark.sql import functions as F
from scipy.stats import spearmanr

from cdm.data import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from cdm.eval import MIN_GROUP
from cdm.explain import grad_cam, last_conv_block, randomized_copy
from cdm.review import ENSEMBLE, confusion_by, flagged, select
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
REVIEW = Path(dbutils.widgets.get("review_root")) / JOB_RUN
SEEDS = ("v2-final-seed0", "v2-final-seed1", "v2-final-seed2")

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if (OUT / "results.json").exists():
    raise RuntimeError(f"{OUT / 'results.json'} exists: Phase D already ran")

pred = pd.read_csv(
    REPO / "results/phase2/analysis_look/predictions.csv.gz", dtype={"skin_type": str}
)

# COMMAND ----------

# Client test rows in the drift run's order (silver, labelled client test, by source and ID).
order = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split = 'test' AND label IS NOT NULL")
    .select("source", "isic_id")
    .orderBy("source", "isic_id")
    .toPandas()
)
test = order.merge(pred, on=["source", "isic_id"], how="left", validate="one_to_one")
assert test[f"{ENSEMBLE}__mel"].notna().all() and len(test) == 3262, len(test)
ood = []
for k in SEEDS:
    with np.load(Path(dbutils.widgets.get("effnet_scores")) / f"{k}_scores.npz") as z:
        ood.append(z["test__mahalanobis"])
ood_scores = np.mean(ood, axis=0)
assert len(ood_scores) == len(test)
chosen = select(test, ood_scores)
print(f"{len(chosen)} images chosen", flush=True)

# COMMAND ----------

meta = (
    spark.read.table("workspace.cdm.silver_images")
    .where(F.col("isic_id").isin(list(chosen["isic_id"])))
    .select("source", "isic_id", "image")
    .toPandas()
    .merge(
        spark.sql(
            "SELECT image_source_value AS isic_id, license, attribution "
            "FROM workspace.cdm.omop_image_occurrence"
        ).toPandas(),
        on="isic_id",
        how="left",
    )
)
chosen = chosen.merge(meta, on=["source", "isic_id"], how="left", validate="one_to_one")
assert chosen["image"].notna().all()

manifest = {m["seed"]: m for m in json.loads(Path(dbutils.widgets.get("models_file")).read_text())}
models = {}
for s, key in enumerate(SEEDS):
    model = build_model(pretrained=False, arch="b0")
    model.load_state_dict(torch.load(manifest[s]["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == manifest[s]["weights_sha256"], f"{key} checkpoint mismatch"
    models[key] = model.eval()

transform = eval_transform(224)
images = torch.stack([transform(Image.open(io.BytesIO(b)).convert("RGB")) for b in chosen["image"]])

# The gate: seed 0 reproduces its committed probabilities on these images.
with torch.no_grad():
    probs0 = torch.softmax(models[SEEDS[0]](images), dim=1).numpy()
committed = chosen[["isic_id"]].merge(pred, on="isic_id")[[f"{SEEDS[0]}__{c}" for c in CLASSES]]
gap = float(np.abs(probs0 - committed.to_numpy()).max())
if gap > 1e-3:
    raise RuntimeError(f"seed 0 differs from its committed probabilities by {gap}")
print(f"reproduction gate passed (largest gap {gap:.2e})", flush=True)

# COMMAND ----------

index = {c: i for i, c in enumerate(CLASSES)}
predicted = torch.tensor([index[c] for c in chosen["predicted"]])
true = torch.tensor([index[c] for c in chosen["label"]])
maps, randomization, agreement = {}, [], []
random0 = randomized_copy(models[SEEDS[0]], seed=0)
for b in range(0, len(images), 16):
    x, p, t = images[b : b + 16], predicted[b : b + 16], true[b : b + 16]
    m0 = grad_cam(models[SEEDS[0]], last_conv_block(models[SEEDS[0]]), x, p)
    maps.setdefault("predicted", []).append(m0)
    maps.setdefault("true", []).append(
        grad_cam(models[SEEDS[0]], last_conv_block(models[SEEDS[0]]), x, t)
    )
    mr = grad_cam(random0, last_conv_block(random0), x, p)
    others = [grad_cam(models[k], last_conv_block(models[k]), x, p) for k in SEEDS[1:]]
    for i in range(len(x)):
        a = m0[i].flatten().numpy()
        randomization.append(float(spearmanr(a, mr[i].flatten().numpy()).statistic))
        agreement.append(
            float(np.mean([spearmanr(a, o[i].flatten().numpy()).statistic for o in others]))
        )
maps = {k: torch.cat(v).numpy() for k, v in maps.items()}
chosen["randomization_spearman"] = randomization
chosen["seed_agreement_spearman"] = agreement
chosen["flagged_not_interpreted"] = [flagged(r) for r in randomization]

# COMMAND ----------

# Overlays: the 224 px crop the model saw, with each map on top.


def jet(x: np.ndarray) -> np.ndarray:
    """The jet colormap: blue (0) through green to red (1)."""
    return np.clip(1.5 - np.abs(4 * x[..., None] - np.array([3.0, 2.0, 1.0])), 0, 1)


mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
REVIEW.mkdir(parents=True, exist_ok=True)


def overlay(i: int, which: str) -> bytes:
    rgb = np.clip(images[i].numpy() * std + mean, 0, 1).transpose(1, 2, 0)
    mixed = 0.6 * rgb + 0.4 * jet(maps[which][i])
    buf = io.BytesIO()
    Image.fromarray((mixed * 255).astype(np.uint8)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


paths, cells = [], []
for i, row in chosen.reset_index(drop=True).iterrows():
    pics = {w: overlay(i, w) for w in ("predicted", "true")}
    for w, data in pics.items():
        (REVIEW / f"{row.isic_id}_{w}.jpg").write_bytes(data)
    paths.append(str(REVIEW / f"{row.isic_id}_predicted.jpg"))
    img = "".join(
        f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(d).decode()}">'
        f"<figcaption>{w} class</figcaption></figure>"
        for w, d in pics.items()
    )
    flag = (
        "<b>flagged: map barely depends on the weights, not interpreted</b><br>"
        if row.flagged_not_interpreted
        else ""
    )
    cells.append(
        f"<section><h3>{html.escape(row.isic_id)}</h3>{img}<p>{flag}"
        f"reason: {html.escape(row.reason)}<br>true: {row.label}, predicted: {row.predicted} "
        f"(p true {row.p_true:.3f}, p predicted {row.p_predicted:.3f})<br>"
        f"Mahalanobis (mean of 3 seeds): {row.ood_score:.1f}<br>"
        f"randomization Spearman {row.randomization_spearman:.2f}, seed agreement {row.seed_agreement_spearman:.2f}<br>"
        f"license: {html.escape(str(row.license))}; attribution: {html.escape(str(row.attribution))}<br>"
        f"reviewer notes: ____________________</p></section>"
    )
chosen["overlay_path"] = paths
page = (
    "<html><head><meta charset='utf-8'><title>Review worksheet</title><style>"
    "section{border-top:1px solid #ccc;padding:8px}figure{display:inline-block;margin:4px}"
    "</style></head><body><h1>Grad-CAM review worksheet (research Phase D)</h1>"
    f"<p>Shipped EfficientNet-B0 v2, seed 0; code {CODE}; job run {JOB_RUN}. Research prototype, "
    "not for clinical use. Maps are 7x7 upsampled: coarse. Images from the ISIC Archive under "
    "the license shown with each.</p>" + "".join(cells) + "</body></html>"
)

# COMMAND ----------

mlflow.set_experiment(experiment_id=dbutils.widgets.get("experiment_id"))
with mlflow.start_run(
    run_name="research/d_review/worksheet",
    tags={"code_version": CODE, "databricks_job_run": JOB_RUN},
) as run:
    mlflow.log_text(page, "review_worksheet.html")
    worksheet_run = run.info.run_id

table = chosen.drop(columns=["image"]).assign(
    reviewer_notes=None, code_version=CODE, job_run_id=JOB_RUN
)
spark.createDataFrame(table.astype({"reviewer_notes": "string"})).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable("workspace.cdm.gold_review_worksheet")

repo_cols = ["isic_id", "client", "label", "reason", "predicted", "p_true", "p_predicted",
             "ood_score", "randomization_spearman", "seed_agreement_spearman",
             "flagged_not_interpreted", "license"]  # fmt: skip
csv_text = chosen[repo_cols].to_csv(index=False, lineterminator="\n", float_format="%.6g")
labelled = pred[pred["label"].notna()]
results = {
    "spec": "results/research/d_review/spec.json",
    "spec_sha256": expected["results/research/d_review/spec.json"],
    "code_version": CODE,
    "wheel_sha256": WHEEL_SHA,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "reproduction_gate_largest_gap": gap,
    "images": int(len(chosen)),
    "by_reason": {
        r: int(chosen["reason"].str.contains(r).sum())
        for r in [f"worst_miss:{c}" for c in CLASSES] + ["most_ood"]
    },
    "randomization_spearman": {
        "mean": float(np.mean(randomization)),
        "min": float(np.min(randomization)),
        "max": float(np.max(randomization)),
    },
    "flagged_not_interpreted": int(chosen["flagged_not_interpreted"].sum()),
    "seed_agreement_spearman": {
        "mean": float(np.mean(agreement)),
        "min": float(np.min(agreement)),
        "max": float(np.max(agreement)),
    },
    "worksheet": {
        "mlflow_run": worksheet_run,
        "artifact": "review_worksheet.html",
        "table": "workspace.cdm.gold_review_worksheet",
        "overlays": str(REVIEW),
    },
    "worksheet_csv_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
    "error_analysis": {
        "confusion_by_client_test": confusion_by(labelled[labelled["split"] == "test"], "client"),
        "confusion_by_skin_type": {
            site: confusion_by(labelled[labelled["source"] == site], "skin_type", MIN_GROUP)
            for site in ("buenos_aires", "pad_ufes")
        },
    },
    "status": "not reviewed: reviewer_notes is blank until a person fills it in",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "worksheet.csv").write_text(csv_text, encoding="utf-8")
(OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: results[k] for k in ("images", "flagged_not_interpreted", "randomization_spearman")}
    )
)
