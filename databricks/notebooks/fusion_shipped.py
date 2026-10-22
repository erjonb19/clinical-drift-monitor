# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase E follow-up: the shipped model's own features, with and without metadata
# MAGIC Owner's decision (2026-10-22, option b): two new variants, image only and image plus
# MAGIC metadata, on the shipped EfficientNet-B0 v2's penultimate features (one checkpoint per
# MAGIC seed; probe seed s trains on checkpoint s's features), with the same metadata encoding and
# MAGIC the same head as Phase E. `mode=select` chooses each variant's settings on validation and
# MAGIC writes `select_shipped.json`, committed as `spec_shipped.json`. `mode=look` is this variant
# MAGIC pair's own single test look: attempt logged, everything in memory, results then lock.
# MAGIC
# MAGIC Caveat stated in the spec: these features come from networks trained on the train split,
# MAGIC so a head on them sees training images the network already fits.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "embeddings": "/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings",
    "features": "/Volumes/workspace/cdm/raw/results/research/e_fusion/shipped_features",
    "out": "/Volumes/workspace/cdm/raw/results/research/e_fusion/shipped",
    "mode": "select",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cdm.fusion import (
    SHIPPED,
    SHIPPED_VARIANTS,
    MetadataEncoder,
    check_order,
    inputs,
    select_per_seed,
)
from cdm.probe import SEEDS, ProbeConfig, train_probe
from cdm.splits import CLASSES

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
FEATS = Path(dbutils.widgets.get("features"))
MODE = dbutils.widgets.get("mode")
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
SPEC = REPO / "results/research/e_fusion/spec_shipped.json"
assert MODE in ("select", "look"), MODE

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if MODE == "look":
    assert "results/research/e_fusion/spec_shipped.json" in expected, (
        "the look needs the committed spec"
    )
    for name in ("results.json", "test_look.json", "results.json.tmp", "test_look.json.tmp"):
        if (OUT / name).exists():
            raise RuntimeError(f"{OUT / name} exists: the test look already happened or began")
elif (OUT / "select_shipped.json").exists():
    raise RuntimeError(f"{OUT / 'select_shipped.json'} exists: selection already ran")

summary = json.loads((FEATS / "summary.json").read_text())
got = hashlib.sha256((FEATS / "features.npz").read_bytes()).hexdigest()
assert got == summary["sha256"]["features.npz"], f"features.npz sha256 {got}"

spec = importlib.util.spec_from_file_location("vlm_baseline", REPO / "scripts" / "vlm_baseline.py")
assert spec is not None and spec.loader is not None
vb = importlib.util.module_from_spec(spec)
sys.modules["vlm_baseline"] = vb  # dataclasses look their module up
spec.loader.exec_module(vb)


def checked(fn, *args):
    """The script refuses with sys.exit, which a notebook may not treat as a failure."""
    try:
        return fn(*args)
    except SystemExit as refusal:
        raise RuntimeError(str(refusal)) from None


emb = checked(vb.Embeddings, Path(dbutils.widgets.get("embeddings")))
SETS = ("train", "val", "test", "buenos_aires", "pad_ufes")

# COMMAND ----------

# Inputs, row for row with the BiomedCLIP embeddings (the extraction job used their order).
attrs = spark.sql(
    """
    SELECT s.source, s.isic_id, m.age_approx, p.gender_concept_id, i.anatomic_site_concept_id
    FROM workspace.cdm.silver_images s
    LEFT JOIN workspace.cdm.silver_metadata m USING (source, isic_id)
    LEFT JOIN workspace.cdm.omop_image_occurrence i ON i.image_source_value = s.isic_id
    LEFT JOIN workspace.cdm.omop_person p ON p.person_id = i.person_id
    """
).toPandas()
assert not attrs.duplicated(["source", "isic_id"]).any(), "an image joined to two rows"
attrs = attrs.set_index(["source", "isic_id"])

rows, meta, labels = {}, {}, {}
with np.load(FEATS / "features.npz") as z:
    shipped = {s: {seed: z[f"seed{seed}__{s}"] for seed in SEEDS} for s in SETS}
for s in SETS:
    r = emb.rows(s)
    rows[s] = r
    meta[s] = attrs.loc[list(zip(r["source"], r["isic_id"], strict=True))].reset_index()
    check_order(list(r["isic_id"]), list(meta[s]["isic_id"]), s)
    assert all(len(shipped[s][seed]) == len(r) for seed in SEEDS), s
    labels[s] = emb.labels(s)
encoder = MetadataEncoder().fit(meta["train"])
metadata = {s: encoder.transform(meta[s]) for s in SETS}
# X[variant][set][seed]
X = {
    v: {
        s: {seed: inputs(v, {SHIPPED: shipped[s][seed]}, metadata[s]) for seed in SEEDS}
        for s in SETS
    }
    for v in SHIPPED_VARIANTS
}
data_hash = hashlib.sha256(
    b"".join(
        X[v][s][seed].tobytes()
        for v in SHIPPED_VARIANTS
        for s in ("train", "val")
        for seed in SEEDS
    )
).hexdigest()
print({v: X[v]["train"][0].shape for v in SHIPPED_VARIANTS}, flush=True)

# COMMAND ----------

if MODE == "select":
    chosen, grids = {}, {}
    for v in SHIPPED_VARIANTS:
        cfg, grid = select_per_seed(
            X[v]["train"], labels["train"], X[v]["val"], labels["val"], len(CLASSES)
        )
        chosen[v] = {k: getattr(cfg, k) for k in ("lr", "weight_decay", "epochs", "batch_size")}
        grids[v] = grid
        print(v, chosen[v], round(max(g["mean"] for g in grid), 4), flush=True)
    select = {
        "committed_before_running": True,
        "decided": datetime.now(UTC).date().isoformat(),
        "decision": "owner, 2026-10-22, option (b): add the shipped model's own features as a new variant, with its own spec and a separate single test look",
        "selection_code_version": CODE,
        "selection_job_run": JOB_RUN,
        "wheel_sha256": WHEEL_SHA,
        "features": {
            "job_run_id": summary["job_run_id"],
            "sha256": summary["sha256"],
            "models": summary["models"],
            "test_top_probability_gap": summary["test_top_probability_gap"],
        },
        "train_val_inputs_sha256": data_hash,
        "variants": list(SHIPPED_VARIANTS),
        "seed_pairing": "probe seed s trains on shipped checkpoint s's features",
        "metadata": {
            "columns": encoder.columns,
            "rule": "as Phase E: fit on train; the image's source is never an input",
        },
        "chosen": chosen,
        "seeds": list(SEEDS),
        "validation_grid": grids,
        "rule": "per variant, the highest mean validation balanced accuracy over the 3 seed pairs (cdm.fusion.select_per_seed)",
        "test_look": {
            "rows": "client_test, client_test_without_overlap_patients, ham10000_test, buenos_aires, pad_ufes (as in Phase E)",
            "bootstrap": "1000 resamples, seed 0, by patient, else lesion, else image",
            "paired": "image + metadata minus image only, same seed, same resamples",
            "reference": "the shipped EfficientNet v2 ensemble from the committed analysis-look predictions",
        },
        "caveat": "the features come from networks trained on the train split, so the head is fit on training images the network already fits; validation-chosen settings guard against this only partly",
    }
    vb.write_json(OUT / "select_shipped.json", select)
    print(f"wrote {OUT / 'select_shipped.json'}")

# COMMAND ----------

if MODE == "look":
    plan = json.loads(SPEC.read_text(encoding="utf-8"))
    if plan["train_val_inputs_sha256"] != data_hash:
        raise RuntimeError("the inputs differ from those the spec was chosen on")
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with (OUT / "look_attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"started_utc": started, "job_run": JOB_RUN, "code_version": CODE, "mode": MODE}
            )
            + "\n"
        )

    seeds = [f"seed{s}" for s in plan["seeds"]]
    sets = ["test", "buenos_aires", "pad_ufes"]
    logits = {}
    for v in SHIPPED_VARIANTS:
        cfg = ProbeConfig(**plan["chosen"][v])
        for s, name in zip(plan["seeds"], seeds, strict=True):
            head = train_probe(X[v]["train"][s], labels["train"], len(CLASSES), s, cfg)
            logits[f"{v}__{name}"] = {k: head.logits(X[v][k][s]) for k in sets}

    table = {}
    for row, (source, mask) in vb.evaluation_rows(emb, vb.LOOK).items():
        sub = emb.rows(source)[mask]
        y = emb.labels(source)[mask]
        units = vb.clusters(list(sub["patient_id"]), list(sub["lesion_id"]), list(sub["isic_id"]))
        models = {m: z[source][mask] for m, z in logits.items()}
        reference = vb.effnet_logits(sub)[vb.EFFNET_MODELS[0]]
        entry = {
            "images": int(mask.sum()),
            "units": len(set(units)),
            "models": {m: vb.scored(z, y, units) for m, z in models.items()},
            "reference_" + vb.EFFNET_MODELS[0]: vb.scored(reference, y, units),
        }
        entry["over_seeds"] = {
            v: {
                k: vb.summarize([float(entry["models"][f"{v}__{n}"][k]) for n in seeds])
                for k in ("balanced_accuracy", "melanoma_sensitivity")
                if all(entry["models"][f"{v}__{n}"][k] is not None for n in seeds)
            }
            for v in SHIPPED_VARIANTS
        }
        entry["paired_fusion_minus_image_only"] = {
            n: vb.paired_bootstrap(
                models[f"fusion__{SHIPPED}__{n}"],
                models[f"image_only__{SHIPPED}__{n}"],
                y,
                units,
                n=vb.BOOTSTRAP,
            )
            for n in seeds
        }
        table[row] = entry
        print(f"{row}: done", flush=True)

    look = {
        "started_utc": started,
        "mode": MODE,
        "spec_sha256": expected["results/research/e_fusion/spec_shipped.json"],
        "code_version": CODE,
        "wheel_sha256": WHEEL_SHA,
        "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    }
    results = {
        "spec": "results/research/e_fusion/spec_shipped.json",
        **{k: look[k] for k in ("spec_sha256", "code_version", "wheel_sha256", "ran_on")},
        "classification": table,
        "note": "descriptive; research use only; never served",
    }
    vb.write_json(OUT / "results.json", results)
    vb.write_json(
        OUT / "test_look.json",
        {
            **look,
            "results_sha256": vb.sha256(OUT / "results.json"),
            "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    print(f"wrote {OUT / 'results.json'}, then {OUT / 'test_look.json'}")
