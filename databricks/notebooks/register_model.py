# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: register the served model in Unity Catalog, and read the lineage (serverless CPU)
# MAGIC Registers the evaluated serving bundle (manifest SHA-256 checked against
# MAGIC `results/phase3/serving/results.json`) as an MLflow pyfunc model in the Unity Catalog
# MAGIC registry, `workspace.cdm.skin_lesion_classifier`, alias `champion`. The run logs the
# MAGIC clients' training split of `workspace.cdm.silver_images` as its training input (an MLflow
# MAGIC dataset record; Unity Catalog lineage into the model itself is not shown here). Then it reads `system.access.table_lineage` for the chain
# MAGIC bronze -> silver -> gold, OMOP and the audit table, and writes it with the model version.
# MAGIC Item 3's challenger job registers its challenger here too, with `alias` empty (the alias never
# MAGIC moves; `promote_challenger` does that by hand), the volume's bundle results as
# MAGIC `evaluated_results`, and `decision_path`: the version is tagged with the decision file's
# MAGIC SHA-256, the rule's verdict (`promoted_by_rule` or `rejected`), the checked bundle's
# MAGIC manifest SHA-256 and `model_gen`.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "bundle": "",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/lineage",
    "experiment_id": "3252981153928765",
    "job_run_id": "",
    "evaluated_results": "",
    "alias": "champion",
    "decision_path": "",
    "model_gen": "",
    "training_data": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import importlib.metadata as md
import io
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from PIL import Image

import cdm
from cdm.serving import Service

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
BUNDLE = Path(dbutils.widgets.get("bundle"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
NAME = "workspace.cdm.skin_lesion_classifier"

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
for rel, digest in json.loads(dbutils.widgets.get("files_sha256")).items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
EVALUATED = Path(
    dbutils.widgets.get("evaluated_results") or REPO / "results/phase3/serving/results.json"
)
ALIAS = dbutils.widgets.get("alias")
assert ALIAS in ("champion", ""), "alias: champion, or empty to leave every alias where it is"
DECISION = dbutils.widgets.get("decision_path")
MODEL_GEN = dbutils.widgets.get("model_gen")
if DECISION:
    assert not ALIAS, "a challenger is registered without moving the alias (promote_challenger)"
    assert MODEL_GEN, "a challenger is registered with its model_gen tag"
    # Read before anything is registered, so a bad decision file stops the run with no version.
    DECISION_BYTES = Path(DECISION).read_bytes()
    VERDICT = json.loads(DECISION_BYTES)["promoted"]
    assert isinstance(VERDICT, bool), "the decision file's promoted is true or false"
evaluated = json.loads(EVALUATED.read_text(encoding="utf-8"))
manifest_sha = hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest()
assert manifest_sha == evaluated["bundle"]["manifest_sha256"], "not the evaluated bundle"
local = Path(tempfile.mkdtemp()) / "bundle"
shutil.copytree(BUNDLE, local)
Service.load(local)  # every file checked against the manifest


class Classifier(mlflow.pyfunc.PythonModel):
    """The served ensemble: one image's bytes per row in, the API's answer out."""

    def load_context(self, context):  # noqa: ANN001
        from cdm.serving import Service

        self.service = Service.load(Path(context.artifacts["bundle"]))

    def predict(self, context, model_input):  # noqa: ANN001
        rows = [self.service.predict(bytes(b)) for b in model_input["image"]]
        return pd.DataFrame(
            [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows]
        )


buf = io.BytesIO()
Image.new("RGB", (64, 64), (180, 120, 100)).save(buf, format="JPEG")
example = pd.DataFrame({"image": [buf.getvalue()]})
model = Classifier()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_id=dbutils.widgets.get("experiment_id"))
pins = [
    f"{p}=={md.version(p)}"
    for p in ("torch", "torchvision", "numpy", "pandas", "pillow", "scikit-learn")
]
with mlflow.start_run(
    run_name="phase3/register_model",
    tags={
        "code_version": CODE,
        "databricks_job_run": JOB_RUN,
        "bundle_manifest_sha256": manifest_sha,
    },
) as run:
    train = (
        spark.read.table("workspace.cdm.silver_images")
        .where("role = 'client' AND split = 'train' AND label IS NOT NULL")
        .select("source", "isic_id", "label")
    )
    mlflow.log_input(
        mlflow.data.from_spark(
            train, table_name="workspace.cdm.silver_images", name="clients_train_split"
        ),
        context="training",
    )
    model.load_context(type("C", (), {"artifacts": {"bundle": str(local)}})())
    signature = infer_signature(example, model.predict(None, example))
    info = mlflow.pyfunc.log_model(
        name="model", python_model=Classifier(), artifacts={"bundle": str(local)},
        code_paths=[str(Path(cdm.__file__).parent)], pip_requirements=pins, signature=signature,
        input_example=example, registered_model_name=NAME,
    )  # fmt: skip
    run_id = run.info.run_id
client = mlflow.MlflowClient(registry_uri="databricks-uc")
version = info.registered_model_version
if ALIAS:
    client.set_registered_model_alias(NAME, ALIAS, version)
tags = {
    "bundle_manifest_sha256": manifest_sha,
    "evaluated_in": str(EVALUATED.relative_to(REPO))
    if EVALUATED.is_relative_to(REPO)
    else str(EVALUATED),
    "code_version": CODE,
}
if MODEL_GEN:
    tags["model_gen"] = MODEL_GEN
if dbutils.widgets.get("training_data"):
    tags["training_data"] = dbutils.widgets.get("training_data")
if DECISION:
    # The tags promote_challenger and move_champion check (cdm.promotion, cdm.registry).
    tags["decision_sha256"] = hashlib.sha256(DECISION_BYTES).hexdigest()
    tags["decision"] = "promoted_by_rule" if VERDICT else "rejected"
    tags["serving_bundle_sha256"] = manifest_sha
for k, v in tags.items():
    client.set_model_version_tag(NAME, version, k, v)

# COMMAND ----------

edges = spark.sql(
    """
    SELECT DISTINCT source_table_full_name AS source, target_table_full_name AS target
    FROM system.access.table_lineage
    WHERE target_table_full_name LIKE 'workspace.cdm.%' AND source_table_full_name IS NOT NULL
      AND source_table_full_name <> target_table_full_name
    ORDER BY target, source
    """
).toPandas()
out = {
    "registered_model": {
        "name": NAME,
        "version": int(version),
        "alias": ALIAS or None,
        "uri": f"models:/{NAME}/{version}",
        "tags": tags,
        "mlflow_run": run_id,
        "training_input": "workspace.cdm.silver_images (clients' training split)",
        "bundle_manifest_sha256": manifest_sha,
    },  # fmt: skip
    "table_lineage_edges": json.loads(edges.to_json(orient="records")),
    "read_from": "system.access.table_lineage",
    "code_version": CODE,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "lineage.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["registered_model"]), len(out["table_lineage_edges"]), "edges")
