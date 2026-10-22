# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase B: BiomedCLIP embeddings (one GPU job)
# MAGIC Embeds every image the Phase 2 drift table used, once, with BiomedCLIP at its pinned
# MAGIC revision: the clients' train, val and test splits, Buenos Aires and PAD-UFES-20 (real
# MAGIC shifts), and the PathMNIST and CIFAR-10 benchmark draws. Also embeds the candidate prompt
# MAGIC templates. No metric is computed here: prompt choice, the linear probe, the committed spec
# MAGIC and the single test look all run afterwards on CPU from these files
# MAGIC (`scripts/vlm_baseline.py`). The model comes from a Hugging Face cache in the volume,
# MAGIC checked by SHA-256 and by pinned revision, and is loaded offline. A deadline stops the job
# MAGIC before the owner's GPU cap (2 hours in total) could be reached.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "model_zip": "/Volumes/workspace/cdm/raw/deploy/biomedclip/hf-cache-9f341de.zip",
    "model_zip_sha256": "",
    "benchmarks": "/Volumes/workspace/cdm/raw/benchmarks/benchmarks.zip",
    "benchmarks_sha256": "",
    "deadline_minutes": "60",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

wheel = dbutils.widgets.get("wheel")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", f"{wheel}[vlm]"], check=True)

# COMMAND ----------

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

START = time.time()
DEADLINE = START + 60 * float(dbutils.widgets.get("deadline_minutes"))
CODE = dbutils.widgets.get("code_version")
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
OUT = Path("/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings")
OUT.mkdir(parents=True, exist_ok=True)

# The model cache: copied to local disk, checked against the SHA-256 given to the job, then
# unpacked, and loaded with Hugging Face offline so nothing else can be fetched.
local = Path(tempfile.gettempdir())
blob = Path(dbutils.widgets.get("model_zip")).read_bytes()
got = hashlib.sha256(blob).hexdigest()
assert got == dbutils.widgets.get("model_zip_sha256"), f"model zip sha256 {got}"
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    z.extractall(local / "hf")
del blob
os.environ["HF_HOME"] = str(local / "hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.vlm import MODEL_VERSION, check_cache, image_embeddings, load, logit_scale, text_embeddings

check_cache(local / "hf")
device = torch.device("cuda")
model, preprocess, tokenizer = load(device)
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

# COMMAND ----------

# The same sets as the Phase 2 drift table (databricks/notebooks/drift_table.py).
blob = Path(dbutils.widgets.get("benchmarks")).read_bytes()
got = hashlib.sha256(blob).hexdigest()
assert got == dbutils.widgets.get("benchmarks_sha256"), f"benchmarks.zip sha256 {got}"
bench: dict[str, list[bytes]] = {}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for row in csv.DictReader(io.StringIO(z.read("index.csv").decode())):
        data = z.read(row["path"])
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        bench.setdefault(f"{row['dataset']}_{row['use']}", []).append(data)
del blob

cols = ["isic_id", "source", "role", "client", "split", "label", "lesion_id", "patient_id", "image"]
everything = (
    spark.read.table("workspace.cdm.silver_images")
    .select(*cols)
    .orderBy("source", "isic_id")
    .toPandas()
)
clients = everything[(everything["role"] == "client") & everything["label"].notna()]
SETS = {
    "train": clients[clients["split"] == "train"],
    "val": clients[clients["split"] == "val"],
    "test": clients[clients["split"] == "test"],
    "buenos_aires": everything[everything["source"] == "buenos_aires"],
    "pad_ufes": everything[everything["source"] == "pad_ufes"],
}
IMAGES = {k: list(v["image"]) for k, v in SETS.items()}
for name in ("pathmnist_eval", "cifar10_eval", "pathmnist_select", "cifar10_select"):
    IMAGES[name] = bench[name]
SIZES = {k: len(v) for k, v in IMAGES.items()}
print(SIZES, flush=True)
meta = [frame.drop(columns="image").assign(set=name) for name, frame in SETS.items()]
del bench

# COMMAND ----------

seconds: dict[str, float] = {}
emb: dict[str, np.ndarray] = {}
for name, images in IMAGES.items():
    if time.time() > DEADLINE:
        raise RuntimeError(f"deadline reached before {name}; embedded so far: {sorted(emb)}")
    t0 = time.perf_counter()
    data = EncodedImages(images, [-1] * len(images), preprocess)
    batches = (x for x, _ in DataLoader(data, batch_size=128, num_workers=WORKERS, pin_memory=True))
    emb[name] = image_embeddings(model, batches, device)
    assert emb[name].shape == (len(images), emb[name].shape[1]), name
    seconds[name] = round(time.perf_counter() - t0, 1)
    print(f"{name}: {emb[name].shape} in {seconds[name]} s", flush=True)
text = text_embeddings(model, tokenizer, device)

# COMMAND ----------

# Volumes do not support the seeks a zip writer needs, so write locally, read back, then copy.
import pandas as pd

files = {"embeddings.npz": emb, "text.npz": {"text": text}}
for fname, arrays in files.items():
    path = local / fname
    np.savez_compressed(path, **arrays)
    with np.load(path) as z:
        assert sorted(z.files) == sorted(arrays), f"{fname} did not read back"
    (OUT / fname).write_bytes(path.read_bytes())
pd.concat(meta).to_csv(OUT / "meta.csv", index=False)

summary = {
    "model_version": MODEL_VERSION,
    "code_version": CODE,
    "job_run_id": JOB_RUN,
    "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "images": SIZES,
    "embedding_dim": int(emb["train"].shape[1]),
    "text_shape": list(text.shape),
    "logit_scale": logit_scale(model),
    "classes": list(CLASSES),
    "seconds_per_set": seconds,
    "seconds_total": round(time.time() - START),
    "sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in [*files, "meta.csv"]},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

# The silver images' embeddings as a table too, for SQL and Phase E (benchmarks stay in files).
rows = pd.concat(
    [
        m[["isic_id", "source"]].assign(features=list(emb[name]))
        for m, name in zip(meta, SETS, strict=True)
    ]
)
(
    spark.createDataFrame(
        rows.assign(model_version=MODEL_VERSION, features=[f.tolist() for f in rows["features"]])
    )
    .write.mode("overwrite")
    .saveAsTable("workspace.cdm.gold_vlm_embeddings")
)
print("VLM_EMBED_SUMMARY " + json.dumps(summary), flush=True)
dbutils.notebook.exit(json.dumps(summary))
