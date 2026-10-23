# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: load the registered model back and run it (serverless CPU)
# MAGIC Loads `models:/workspace.cdm.skin_lesion_classifier@champion` in a fresh environment and
# MAGIC predicts made-up images, then compares with the evaluated bundle run in-process through
# MAGIC `cdm.serving.Service`. Writes `registered_model_check.json`.

# COMMAND ----------

for name, default in {
    "bundle": "",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/lineage",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from PIL import Image

from cdm.serving import Service

mlflow.set_registry_uri("databricks-uc")
URI = "models:/workspace.cdm.skin_lesion_classifier@champion"
images = []
for colour in ((180, 120, 100), (90, 60, 50), (200, 170, 150)):
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(buf, format="JPEG")
    images.append(buf.getvalue())
loaded = mlflow.pyfunc.load_model(URI)
out = loaded.predict(pd.DataFrame({"image": images}))
local = Service.load(Path(dbutils.widgets.get("bundle")))
gaps = []
for i, img in enumerate(images):
    ref = local.predict(img)
    got = json.loads(out.iloc[i]["probabilities"])
    gaps.append(max(abs(got[c] - ref["probabilities"][c]) for c in ref["probabilities"]))
    assert (
        out.iloc[i]["label"] == ref["label"]
        and abs(out.iloc[i]["ood_score"] - ref["ood_score"]) < 1e-3
    )
version = (
    mlflow.MlflowClient(registry_uri="databricks-uc")
    .get_model_version_by_alias("workspace.cdm.skin_lesion_classifier", "champion")
    .version
)
result = {
    "uri": URI,
    "version": int(version),
    "images": len(images),
    "labels_match": True,
    "max_probability_gap": float(np.max(gaps)),
    "loaded_in": "a fresh serverless CPU environment (pip requirements from the model)",
    "job_run": dbutils.widgets.get("job_run_id") or "manual",
    "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
}
Path(dbutils.widgets.get("out")).mkdir(parents=True, exist_ok=True)
(Path(dbutils.widgets.get("out")) / "registered_model_check.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
