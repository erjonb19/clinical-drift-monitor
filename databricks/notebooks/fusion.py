# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase E: image plus metadata (serverless CPU)
# MAGIC The same linear head (`cdm.probe`) on the same frozen embeddings, with and without
# MAGIC metadata (age, sex and body site as OMOP concept IDs, with missing-value indicators;
# MAGIC encoding fit on train). Embeddings: BiomedCLIP (Phase B) and the ImageNet-pretrained
# MAGIC EfficientNet-B0 in `gold_embeddings` (not the shipped model's own features).
# MAGIC
# MAGIC `mode=select` chooses each variant's learning rate and weight decay on validation and
# MAGIC writes `select.json`; the owner's repository commits it as the spec. `mode=look` is the one
# MAGIC test look under that committed spec: it logs the attempt, computes everything in memory,
# MAGIC writes `results.json`, and only then the lock `test_look.json`.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "embeddings": "/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings",
    "out": "/Volumes/workspace/cdm/raw/results/research/e_fusion",
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

from cdm.fusion import FEATURE_SETS, VARIANTS, MetadataEncoder, check_order, inputs
from cdm.probe import SEEDS, ProbeConfig, select_config, train_probe
from cdm.splits import CLASSES

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
MODE = dbutils.widgets.get("mode")
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
SPEC = REPO / "results/research/e_fusion/spec.json"
assert MODE in ("select", "look"), MODE

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if MODE == "look":
    assert "results/research/e_fusion/spec.json" in expected, "the look needs the committed spec"
    for name in ("results.json", "test_look.json", "results.json.tmp", "test_look.json.tmp"):
        if (OUT / name).exists():
            raise RuntimeError(f"{OUT / name} exists: the test look already happened or began")
elif (OUT / "select.json").exists():
    raise RuntimeError(f"{OUT / 'select.json'} exists: selection already ran")

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

# Inputs for every set, row for row with the BiomedCLIP embeddings.
gold = spark.sql(
    "SELECT source, isic_id, features FROM workspace.cdm.gold_embeddings "
    "WHERE model_version = 'imagenet-effb0'"
).toPandas()
assert not gold.duplicated(["source", "isic_id"]).any()
gold = gold.set_index(["source", "isic_id"])
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

rows, images, meta, labels = {}, {}, {}, {}
for s in SETS:
    r = emb.rows(s)
    keys = list(zip(r["source"], r["isic_id"], strict=True))
    rows[s] = r
    effb0 = np.stack(gold.loc[keys, "features"].to_numpy()).astype(np.float32)
    images[s] = {"biomedclip": emb.feats[s].astype(np.float32), "imagenet_effb0": effb0}
    meta[s] = attrs.loc[keys].reset_index()
    check_order(list(r["isic_id"]), list(meta[s]["isic_id"]), s)
    labels[s] = emb.labels(s)
encoder = MetadataEncoder().fit(meta["train"])
metadata = {s: encoder.transform(meta[s]) for s in SETS}
X = {v: {s: inputs(v, images[s], metadata[s]) for s in SETS} for v in VARIANTS}
data_hash = hashlib.sha256(
    b"".join(X[v][s].tobytes() for v in VARIANTS for s in ("train", "val"))
).hexdigest()
print({v: X[v]["train"].shape for v in VARIANTS}, flush=True)

# COMMAND ----------

if MODE == "select":
    chosen, grids = {}, {}
    for v in VARIANTS:
        cfg, grid = select_config(
            X[v]["train"], labels["train"], X[v]["val"], labels["val"], len(CLASSES)
        )
        chosen[v] = {k: getattr(cfg, k) for k in ("lr", "weight_decay", "epochs", "batch_size")}
        grids[v] = grid
        print(v, chosen[v], round(max(g["mean"] for g in grid), 4), flush=True)
    select = {
        "committed_before_running": True,
        "decided": datetime.now(UTC).date().isoformat(),
        "selection_code_version": CODE,
        "selection_job_run": JOB_RUN,
        "wheel_sha256": WHEEL_SHA,
        "train_val_inputs_sha256": data_hash,
        "variants": list(VARIANTS),
        "feature_sets": {
            "biomedclip": f"Phase B embeddings (job run {emb.summary['job_run_id']})",
            "imagenet_effb0": "workspace.cdm.gold_embeddings, model_version imagenet-effb0 (ImageNet weights, not the shipped model)",
        },
        "metadata": {
            "columns": encoder.columns,
            "age_mean": encoder.age_mean,
            "age_std": encoder.age_std,
            "rule": "fit on train; unseen categories are unknown; the image's source is never an input",
        },
        "chosen": chosen,
        "seeds": list(SEEDS),
        "validation_grid": grids,
        "rule": "per variant, the highest mean validation balanced accuracy over the 3 seeds (cdm.probe.select_config)",
        "test_look": {
            "rows": "client_test, client_test_without_overlap_patients, ham10000_test, buenos_aires, pad_ufes (as in Phase B)",
            "bootstrap": "1000 resamples, seed 0, by patient, else lesion, else image",
            "paired": "fusion minus image only, same feature set and seed, on the same resamples",
            "source_held_out": "per client source, train without it (encoder refit) and score its test images, image only and fusion, 3 seeds, point estimates",
            "reference": "the shipped EfficientNet v2 ensemble from the committed analysis-look predictions",
        },
        "risks": "metadata can encode the source and so label prevalence; small gains sit inside the intervals",
    }
    vb.write_json(OUT / "select.json", select)
    print(f"wrote {OUT / 'select.json'}")

# COMMAND ----------

if MODE == "look":
    plan = json.loads(SPEC.read_text(encoding="utf-8"))
    if plan["train_val_inputs_sha256"] != data_hash:
        raise RuntimeError("the inputs differ from those the spec was chosen on")
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with (OUT / "look_attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"started_utc": started, "job_run": JOB_RUN, "code_version": CODE}) + "\n"
        )

    seeds = [f"seed{s}" for s in plan["seeds"]]
    sets = ["test", "buenos_aires", "pad_ufes"]
    logits = {}
    for v in VARIANTS:
        cfg = ProbeConfig(**plan["chosen"][v])
        for s, name in zip(plan["seeds"], seeds, strict=True):
            head = train_probe(X[v]["train"], labels["train"], len(CLASSES), s, cfg)
            logits[f"{v}__{name}"] = {k: head.logits(X[v][k]) for k in sets}

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
            for v in VARIANTS
        }
        entry["paired_fusion_minus_image_only"] = {
            f"{f}__{n}": vb.paired_bootstrap(
                models[f"fusion__{f}__{n}"],
                models[f"image_only__{f}__{n}"],
                y,
                units,
                n=vb.BOOTSTRAP,
            )
            for f in FEATURE_SETS
            for n in seeds
        }
        table[row] = entry
        print(f"{row}: done", flush=True)

    held_out = {}
    train_sources = sorted(set(rows["train"]["source"]))
    for src in train_sources:
        keep = (rows["train"]["source"] != src).to_numpy()
        target = (rows["test"]["source"] == src).to_numpy()
        enc = MetadataEncoder().fit(meta["train"][keep])
        m_train, m_test = enc.transform(meta["train"][keep]), enc.transform(meta["test"][target])
        out = {"test_images": int(target.sum())}
        for f in FEATURE_SETS:
            for kind in ("image_only", "fusion"):
                v = f"{kind}__{f}"
                x_tr = inputs(v, {f: images["train"][f][keep]}, m_train)
                x_te = inputs(v, {f: images["test"][f][target]}, m_test)
                cfg = ProbeConfig(**plan["chosen"][v])
                bal = []
                for s in plan["seeds"]:
                    head = train_probe(x_tr, labels["train"][keep], len(CLASSES), s, cfg)
                    bal.append(
                        float(
                            vb.classification_metrics(labels["test"][target], head.logits(x_te))[
                                "balanced_accuracy"
                            ]
                        )
                    )
                out[v] = vb.summarize(bal)
        held_out[src] = out
        print(f"held out {src}: done", flush=True)

    look = {
        "started_utc": started,
        "spec_sha256": expected["results/research/e_fusion/spec.json"],
        "code_version": CODE,
        "wheel_sha256": WHEEL_SHA,
        "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    }
    results = {
        "spec": "results/research/e_fusion/spec.json",
        **{k: look[k] for k in ("spec_sha256", "code_version", "wheel_sha256", "ran_on")},
        "classification": table,
        "source_held_out": held_out,
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
