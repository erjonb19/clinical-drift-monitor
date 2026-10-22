# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase C: subgroups (serverless CPU)
# MAGIC Performance and drift by age band, sex and body site, for the shipped EfficientNet
# MAGIC ensemble and BiomedCLIP (probe and zero-shot), under `results/research/c_subgroups/spec.json`
# MAGIC as amended by `spec_v2.json` (the MSK-overlap row, zero-shot, alert rates by group, intervals
# MAGIC only for reported metrics, and a check that saved drift scores line up with the images).
# MAGIC Sex and body site come from the OMOP tables as concept IDs; age from silver.
# MAGIC
# MAGIC No new model and no new choice: EfficientNet's predictions are the committed analysis-look
# MAGIC file, and the probe is regenerated from the Phase B spec. Before any subgroup is computed,
# MAGIC every model's overall balanced accuracy must equal the committed Phase B results, or the
# MAGIC run stops. Drift reuses the Phase 2 drift run's saved per-image scores.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "embeddings": "/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings",
    "effnet_scores": "/Volumes/workspace/cdm/raw/results/phase2/drift/v2",
    "out": "/Volumes/workspace/cdm/raw/results/research/c_subgroups",
    "bootstrap": "1000",
    "result_name": "results_v2.json",
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
import pandas as pd

from cdm.ood import MahalanobisDetector
from cdm.probe import ProbeConfig, train_probe
from cdm.splits import CLASSES
from cdm.subgroups import (
    age_band,
    alert_rate,
    check_aligned,
    check_reproduced,
    concept_group,
    drift,
    performance,
)
from cdm.vlm import clusters

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
N_BOOT = int(dbutils.widgets.get("bootstrap"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
RESULT = OUT / dbutils.widgets.get("result_name")
SITES = ("buenos_aires", "pad_ufes")
ENS = "v2-ensemble-final-seeds0-1-2"
EFFNET_SEEDS = ("v2-final-seed0", "v2-final-seed1", "v2-final-seed2")

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if RESULT.exists():
    raise RuntimeError(f"{RESULT} exists: this Phase C run already happened")
# silver_images is a view (no Delta history): fingerprint the image list instead.
_ids = spark.sql(
    "SELECT source, isic_id FROM workspace.cdm.silver_images ORDER BY source, isic_id"
).toPandas()
SILVER_FINGERPRINT = {
    "images": len(_ids),
    "sha256_of_ordered_source_isic_id": hashlib.sha256(
        "\n".join(_ids["source"] + ":" + _ids["isic_id"]).encode()
    ).hexdigest(),
}

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
spec_b = json.loads(vb.SPEC.read_text(encoding="utf-8"))
checked(vb.check_inputs, emb, spec_b, vb.LOOK)
committed_b = json.loads((vb.OUT / "results.json").read_text(encoding="utf-8"))

# COMMAND ----------

# Group values for every image of the populations, by ISIC ID.
attrs = spark.sql(
    """
    SELECT s.source, s.isic_id, s.client, m.age_approx, m.fitzpatrick_skin_type,
           p.gender_concept_id, i.anatomic_site_concept_id
    FROM workspace.cdm.silver_images s
    LEFT JOIN workspace.cdm.silver_metadata m USING (source, isic_id)
    LEFT JOIN workspace.cdm.omop_image_occurrence i ON i.image_source_value = s.isic_id
    LEFT JOIN workspace.cdm.omop_person p ON p.person_id = i.person_id
    """
).toPandas()
assert not attrs.duplicated(["source", "isic_id"]).any(), "an image joined to two rows"
attrs = attrs.set_index(["source", "isic_id"])


def groups_of(rows: pd.DataFrame) -> dict[str, list[str]]:
    a = attrs.loc[list(zip(rows["source"], rows["isic_id"], strict=True))]
    return {
        "age_band": [age_band(v) for v in a["age_approx"]],
        "sex": [concept_group(v) for v in a["gender_concept_id"]],
        "body_site": [concept_group(v) for v in a["anatomic_site_concept_id"]],
        "skin_type": [
            v if isinstance(v, str) and v else "not recorded" for v in a["fitzpatrick_skin_type"]
        ],
        "client": [v if isinstance(v, str) and v else "none" for v in a["client"]],
    }


# COMMAND ----------

# The probe, regenerated exactly as in the Phase B look.
cfg = ProbeConfig(**spec_b["probe"]["chosen"])
y_train = emb.labels("train")
seeds = [f"probe_seed{s}" for s in spec_b["probe"]["seeds"]]
sets = ["test", *SITES]
probe = {}
for seed, name in zip(spec_b["probe"]["seeds"], seeds, strict=True):
    head = train_probe(emb.feats["train"], y_train, len(CLASSES), seed, cfg)
    probe[name] = {s: head.logits(emb.feats[s]) for s in sets}
text = vb.class_text(
    emb.text, vb.template_choices(len(vb.TEMPLATES))[spec_b["zero_shot"]["chosen"]]
)
zero_shot = {s: vb.zero_shot_logits(emb.feats[s], text, emb.scale) for s in sets}

populations = {}
for row, (source, mask) in vb.evaluation_rows(emb, vb.LOOK).items():
    if row not in ("client_test", "client_test_without_overlap_patients", *SITES):
        continue
    sub = emb.rows(source)[mask]
    labels = emb.labels(source)[mask]
    logits = {ENS: vb.effnet_logits(sub)[ENS], "zero_shot": zero_shot[source][mask]} | {
        m: probe[m][source][mask] for m in seeds
    }
    for m, z in logits.items():
        bal = vb.classification_metrics(labels, z)["balanced_accuracy"]
        check_reproduced(
            f"{row} {m}", bal, committed_b["classification"][row]["models"][m]["balanced_accuracy"]
        )
    units = clusters(list(sub["patient_id"]), list(sub["lesion_id"]), list(sub["isic_id"]))
    populations[row] = (sub, labels, logits, units)
print("reproduction gate passed for every model and population", flush=True)

# COMMAND ----------

performance_out = {}
for row, (sub, labels, logits, units) in populations.items():
    performance_out[row] = {
        "images": int(len(labels)),
        **{
            attr: performance(labels, logits, units, groups, seeds, n_boot=N_BOOT)
            for attr, groups in groups_of(sub).items()
        },
    }
    print(f"{row}: done", flush=True)

# COMMAND ----------

# Drift: the whole client test split against each group at each external site.
score_files = {
    k: Path(dbutils.widgets.get("effnet_scores")) / f"{k}_scores.npz" for k in EFFNET_SEEDS
}
# The saved scores carry no image IDs: check they line up with the images row for row.
alignment = {}
for k, path in score_files.items():
    with np.load(path) as z:
        for s in ("test", *SITES):
            recomputed = -np.exp(vb.effnet_logits(emb.rows(s))[k].max(axis=1))
            alignment[f"{k}/{s}"] = check_aligned(f"{k} {s}", z[f"{s}__max_softmax"], recomputed)
print(f"alignment gate passed: {alignment}", flush=True)

x_train = emb.feats["train"].astype(np.float64)
maha = MahalanobisDetector(spec_b["drift"]["pca_components_method_d"]).fit(x_train, y_train)
vlm_test = maha.score(emb.feats["test"].astype(np.float64))
drift_out = {}
for site in SITES:
    site_rows = emb.rows(site)
    groups = groups_of(site_rows)
    effnet = {}
    for k, path in score_files.items():
        with np.load(path) as z:
            test, scores = z["test__mahalanobis"], z[f"{site}__mahalanobis"]
        assert len(test) == len(emb.rows("test")) and len(scores) == len(site_rows), (k, site)
        effnet[k] = {attr: drift(test, scores, g) for attr, g in groups.items()}
    vlm_site = maha.score(emb.feats[site].astype(np.float64))
    summary = {}
    for attr in groups:
        summary[attr] = {}
        for group, first in effnet[EFFNET_SEEDS[0]][attr].items():
            if "auroc" in first:
                summary[attr][group] = {
                    "images": first["images"],
                    **{
                        m: vb.summarize([effnet[k][attr][group][m] for k in EFFNET_SEEDS])
                        for m in ("auroc", "fpr95")
                    },
                }
            else:
                summary[attr][group] = first
    drift_out[site] = {
        "efficientnet_mahalanobis_over_seeds": summary,
        "efficientnet_mahalanobis_by_seed": effnet,
        "biomedclip_mahalanobis": {
            attr: drift(vlm_test, vlm_site, g) for attr, g in groups.items()
        },
    }

# Alert rate by group inside the client test split (5% overall by construction).
test_groups = groups_of(emb.rows("test"))
alerts_by_seed = {}
for k, path in score_files.items():
    with np.load(path) as z:
        alerts_by_seed[k] = {
            a: alert_rate(z["test__mahalanobis"], g) for a, g in test_groups.items()
        }
alerts = {"efficientnet_mean_over_seeds": {}, "efficientnet_by_seed": alerts_by_seed,
          "biomedclip": {a: alert_rate(vlm_test, g) for a, g in test_groups.items()}}  # fmt: skip
for a in test_groups:
    first = alerts_by_seed[EFFNET_SEEDS[0]][a]
    alerts["efficientnet_mean_over_seeds"][a] = {
        group: {
            "images": v["images"],
            "alert_rate": vb.summarize(
                [alerts_by_seed[k][a][group]["alert_rate"] for k in EFFNET_SEEDS]
            ),
        }
        if "alert_rate" in v
        else v
        for group, v in first.items()
        if group != "threshold"
    }

# COMMAND ----------

results = {
    "spec": "results/research/c_subgroups/spec_v2.json (amending spec.json)",
    "spec_sha256": expected["results/research/c_subgroups/spec_v2.json"],
    "first_spec_sha256": expected["results/research/c_subgroups/spec.json"],
    "silver_images_fingerprint": SILVER_FINGERPRINT,
    "alignment_gate_mean_gaps": alignment,
    "alert_rate_client_test": alerts,
    "code_version": CODE,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "wheel_sha256": WHEEL_SHA,
    "inputs_sha256": expected
    | {f"effnet_scores/{p.name}": vb.sha256(p) for p in score_files.values()},
    "reproduction_gate": "passed: every model's overall balanced accuracy equals results/research/b_vlm/results.json",
    "performance": performance_out,
    "drift": drift_out,
    "note": "descriptive only; groups differ in class mix, site and imaging type, so no gap is attributed to the attribute",
}
vb.write_json(RESULT, results)
print(f"wrote {RESULT} sha256 {vb.sha256(RESULT)}")
