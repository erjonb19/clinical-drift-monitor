# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 overnight GPU task: control seeds 1 and 2, then FedAvg and FedProx (3 seeds)
# MAGIC One job on one GPU, in this order: control seed 1, control seed 2, then FedAvg and
# MAGIC FedProx alternating by seed (0, 1, 2), so a run cut short still leaves matched pairs.
# MAGIC Every run saves its checkpoint and a result JSON (validation only) to the volume as soon
# MAGIC as it finishes; a run whose result file already exists is skipped, so a retry resumes.
# MAGIC A failed run is recorded in `summary.json` and the next one starts; the task itself
# MAGIC does not fail for it, so the job's one retry is spent only on infrastructure failures
# MAGIC (no GPU, a crashed process). A deadline per job run, counted from the first task
# MAGIC attempt of that job run, caps GPU time across its attempts: no training run starts
# MAGIC after `deadline_minutes`. A later job run gets a fresh deadline and resumes by skipping
# MAGIC finished results. When every planned run has finished, the job IDs in
# MAGIC `pause_when_done` (the later scheduled attempts) have their schedules paused.
# MAGIC Writes the manifests the CPU test task reads: `final_models.json` (control seeds 0-2)
# MAGIC and `federated_models.json`.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "control_seed0": "{}",
    "rounds": "20",
    "proximal_mu": "0.01",
    "deadline_minutes": "100",
    "job_run_id": "",  # the job passes {{job.run_id}}: the same for a task's retries
    "pause_when_done": "",  # comma-separated job IDs
    # A federated run whose image loading fails is retried at once, in the same session,
    # with this many loader workers and pin_memory off; "" turns the retry off.
    "hardened_workers": "2",
    # "" is v1. "v2" writes to overnight/v2 with v2- checkpoints, trains centralized seed 0
    # too, and also scores v1's exact validation images (from the v1 snapshot table).
    "version": "",
    "v1_snapshot": "workspace.cdm.silver_images_v1_snapshot",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel"),
     "flwr==1.39.0"],
    check=True,
)  # fmt: skip

# COMMAND ----------

import gc
import json
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.federated import FedConfig, run_federated
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.report import score_rows
from cdm.silver import EncodedImages, split_datasets
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
CONTROL0 = json.loads(dbutils.widgets.get("control_seed0"))
ROUNDS = int(dbutils.widgets.get("rounds"))
MU = float(dbutils.widgets.get("proximal_mu"))
CODE = dbutils.widgets.get("code_version")
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
VERSION = dbutils.widgets.get("version")
PREFIX = f"{VERSION}-" if VERSION else ""
CKPT = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2")
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/overnight") / VERSION
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
# The deadline counts from the first attempt of this job run that reached this cell, before
# data loading. Keyed by job run, so a new job run is not stopped by an old run's clock.
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
first_start = OUT / f"first_code_start-{JOB_RUN}.txt"
if not first_start.exists():
    first_start.write_text(str(time.time()))
DEADLINE = float(first_start.read_text()) + 60 * float(dbutils.widgets.get("deadline_minutes"))
if time.time() > DEADLINE:
    raise RuntimeError("deadline passed before this attempt started: no runs started")
cfg = TrainConfig(**{"num_workers": WORKERS, **CONFIG})
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/cdm-phase2")

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
pooled = split_datasets(rows, cfg)
clients = {
    c: split_datasets(g.reset_index(drop=True), cfg)["train"] for c, g in rows.groupby("client")
}
val = rows[rows["split"] == "val"].reset_index(drop=True)
print({c: len(d) for c, d in clients.items()}, flush=True)
V1_IDS: set[str] = set()
if VERSION:
    V1_IDS = {
        r[0]
        for r in spark.read.table(dbutils.widgets.get("v1_snapshot")).select("isic_id").collect()
    }
    train_rows = rows[rows["split"] == "train"]
    n_train = len(train_rows)
    (OUT / "data_summary.json").write_text(json.dumps({
        "version": VERSION, "code_version": CODE,
        "images": {s: int((rows["split"] == s).sum()) for s in ("train", "val", "test")},
        "train_images_by_client": {c: int(n) for c, n in train_rows["client"].value_counts().items()},
        # FedAvg weights each client by its training images, so these are also the
        # aggregation weights.
        "train_share_by_client": {c: round(n / n_train, 4)
                                  for c, n in train_rows["client"].value_counts().items()},
        "bordered_share_of_train": round(float(train_rows["bordered"].mean()), 4),
        "bordered_train_by_client": {c: int(g["bordered"].sum()) for c, g in train_rows.groupby("client")},
        "v1_images_in_silver": int(rows["isic_id"].isin(V1_IDS).sum()),
        "v1_val_images": int(val["isic_id"].isin(V1_IDS).sum()),
        "v1_test_images": int(((rows["split"] == "test") & rows["isic_id"].isin(V1_IDS)).sum()),
    }, indent=2))  # fmt: skip

# COMMAND ----------


def val_scores(model: torch.nn.Module) -> dict[str, object]:
    data = EncodedImages(
        list(val["image"]),
        val["label"].map({c: i for i, c in enumerate(CLASSES)}).tolist(),
        eval_transform(cfg.image_size, cfg.color_constancy),
    )
    _, logits, labels = extract(
        model, DataLoader(data, batch_size=128, num_workers=WORKERS), device
    )
    scores = score_rows(val, logits, labels)
    if VERSION:  # the same predictions, restricted to v1's exact validation images
        v1 = val["isic_id"].isin(V1_IDS).to_numpy()
        scores["val_v1_images"] = score_rows(val[v1].reset_index(drop=True), logits[v1],
                                             labels[v1])["val"]  # fmt: skip
    return scores


LOADER_ERRORS = (
    "Pin memory thread exited",
    "DataLoader worker",
    "unable to allocate shared memory",
)


def run(kind: str, seed: int, hardened: bool = False, previous_failure: str = "") -> None:
    name = f"{PREFIX}{kind}-seed{seed}"
    out_file = OUT / f"{name}.json"
    if out_file.exists():
        print(f"{name}: result exists, skipped", flush=True)
        return
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with mlflow.start_run(run_name=f"overnight-{name}") as mlrun:
        mlflow.log_params({"kind": kind, "seed": seed, "rounds": ROUNDS, "mu": MU, **CONFIG})
        generator = seed_everything(seed, cfg.num_threads, gpu=True)
        model = build_model(pretrained=True, arch=cfg.arch)
        t0 = time.perf_counter()
        run_cfg = cfg
        if hardened:
            run_cfg = TrainConfig(
                **{**cfg.__dict__, "num_workers": int(dbutils.widgets.get("hardened_workers"))}
            )
        loader = {"version": "hardened" if hardened else "default",
                  "num_workers": run_cfg.num_workers, "pin_memory": not hardened}  # fmt: skip
        if kind == "final":
            model, history = train(model, pooled["train"], pooled["val"], cfg, seed, device,
                                   generator)  # fmt: skip
            extra: dict[str, object] = {}
        else:
            fcfg = FedConfig(method=kind, rounds=ROUNDS, proximal_mu=MU)
            model, history = run_federated(
                model, clients, pooled["val"], run_cfg, fcfg, seed, device, generator,
                pin_memory=not hardened,
            )  # fmt: skip
            extra = {
                "fed_config": fcfg.__dict__,
                "clients": {c: len(d) for c, d in clients.items()},
            }
        seconds = time.perf_counter() - t0
        scores = val_scores(model)
        folder = CKPT / name
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "/tmp/model.pt")
        (folder / "model.pt").write_bytes(Path("/tmp/model.pt").read_bytes())
        pooled_val = scores["val"]["pooled"]
        mlflow.log_metrics({"val_bacc_pooled": pooled_val["balanced_accuracy"]})
        result = {
            "run": name, "kind": kind, "seed": seed, "code_version": CODE, "config": CONFIG,
            **extra, "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "train_seconds": round(seconds),
            "metrics": {k: pooled_val[k] for k in ("balanced_accuracy", "macro_auroc",
                                                   "melanoma_sensitivity")},
            "scores": scores, "history": history,
            "checkpoint": str(folder / "model.pt"), "weights_sha256": weights_hash(model),
            "mlflow_run_id": mlrun.info.run_id, "gpu": torch.cuda.get_device_name(0),
            "loader": loader, "previous_failure": previous_failure or None,
        }  # fmt: skip
    out_file.write_text(json.dumps(result))
    print(json.dumps({"run": name, "metrics": result["metrics"], "seconds": round(seconds)}),
          flush=True)  # fmt: skip


def manifest(kinds: tuple[str, ...]) -> list[dict[str, object]]:
    found = []
    for f in sorted(OUT.glob("*-seed*.json")):
        r = json.loads(f.read_text())
        if r.get("kind") in kinds:
            found.append({k: r[k] for k in ("kind", "seed", "checkpoint", "weights_sha256")})
    return found


# COMMAND ----------

FINAL_SEEDS = [0, 1, 2] if VERSION else [1, 2]  # v1's seed 0 came from the tuning ladder
PLAN = [("final", s) for s in FINAL_SEEDS] + [
    (k, s) for s in (0, 1, 2) for k in ("fedavg", "fedprox")
]
failed, not_started = {}, []
for kind, seed in PLAN:
    if time.time() > DEADLINE and not (OUT / f"{PREFIX}{kind}-seed{seed}.json").exists():
        not_started.append(f"{PREFIX}{kind}-seed{seed}")
        continue
    try:
        run(kind, seed)
    except Exception:
        error = traceback.format_exc()[-3000:]
        print(f"{PREFIX}{kind}-seed{seed} FAILED\n{error}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        retry = (kind != "final" and dbutils.widgets.get("hardened_workers")
                 and any(e in error for e in LOADER_ERRORS))  # fmt: skip
        if not retry:
            failed[f"{PREFIX}{kind}-seed{seed}"] = error
            continue
        print(
            f"{PREFIX}{kind}-seed{seed}: loader error, retrying with the hardened loader",
            flush=True,
        )
        try:
            run(kind, seed, hardened=True, previous_failure=error)
        except Exception:
            failed[f"{PREFIX}{kind}-seed{seed}"] = (
                error + "\n--- hardened retry ---\n" + traceback.format_exc()[-3000:]
            )
            print(f"{PREFIX}{kind}-seed{seed} hardened retry FAILED", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

finals = ([] if VERSION else [{"kind": "final", "seed": 0, **CONTROL0}]) + manifest(("final",))
(OUT / "final_models.json").write_text(json.dumps(finals))
(OUT / "federated_models.json").write_text(json.dumps(manifest(("fedavg", "fedprox"))))
summary = {"code_version": CODE, "plan": [f"{PREFIX}{k}-seed{s}" for k, s in PLAN],
           "finished": sorted(f.stem for f in OUT.glob("*-seed*.json")), "failed": failed,
           "not_started_deadline": not_started}  # fmt: skip
(OUT / "summary.json").write_text(json.dumps(summary))
pause = [j.strip() for j in dbutils.widgets.get("pause_when_done").split(",") if j.strip()]
if pause and not summary["failed"] and set(summary["plan"]) <= set(summary["finished"]):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.jobs import CronSchedule, JobSettings, PauseStatus

    w = WorkspaceClient()
    for job_id in pause:
        schedule = w.jobs.get(int(job_id)).settings.schedule
        if schedule is not None:
            paused = CronSchedule(quartz_cron_expression=schedule.quartz_cron_expression,
                                  timezone_id=schedule.timezone_id,
                                  pause_status=PauseStatus.PAUSED)  # fmt: skip
            w.jobs.update(int(job_id), new_settings=JobSettings(schedule=paused))
    summary["paused_jobs"] = pause
summary["job_run_id"] = JOB_RUN
(OUT / "summary.json").write_text(json.dumps(summary))
print("OVERNIGHT_SUMMARY " + json.dumps(summary), flush=True)
dbutils.notebook.exit(json.dumps(summary))
