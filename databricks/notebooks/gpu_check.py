# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 GPU check
# MAGIC Is serverless GPU usable from this Free Edition workspace, which GPU is it, how fast does
# MAGIC EfficientNet-B0 train on it, and are two same-seed runs identical? Results are printed as
# MAGIC one JSON line starting with `GPU_CHECK_RESULT` and returned as the notebook's exit value.

# COMMAND ----------

# MAGIC %pip install --quiet /Volumes/workspace/cdm/raw/deploy/c74e24a/cdm-0.1.0-py3-none-any.whl

# COMMAND ----------

import hashlib
import io
import json
import subprocess
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

from cdm.train import build_model

result: dict[str, object] = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    result["gpu"] = props.name
    result["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    result["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(json.dumps(result))

# COMMAND ----------


# Training speed on synthetic 224 x 224 batches: the GPU's compute ceiling.
def train_steps(x: torch.Tensor, y: torch.Tensor, steps: int, seed: int) -> tuple[float, str]:
    torch.manual_seed(seed)
    model = build_model(pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize() if device.type == "cuda" else None
    rate = steps * x.shape[0] / (time.perf_counter() - start)
    digest = hashlib.sha256()
    for t in model.state_dict().values():
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    return rate, digest.hexdigest()[:16]


gen = torch.Generator().manual_seed(0)
x = torch.randn(64, 3, 224, 224, generator=gen).to(device)
y = torch.randint(0, 7, (64,), generator=gen).to(device)
rate, _ = train_steps(x, y, steps=50, seed=0)
result["synthetic_train_images_per_s"] = round(rate)

# Same seed twice, with deterministic algorithms in warn-only mode (EfficientNet's pooling has
# no deterministic CUDA backward): are the weights identical?
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
_, a = train_steps(x, y, steps=20, seed=1)
_, b = train_steps(x, y, steps=20, seed=1)
result["same_seed_identical_weights"] = a == b
print(json.dumps(result))

# COMMAND ----------

# Training speed on real silver images read through Spark, including PNG decoding.
rows = (
    spark.read.table("workspace.cdm.silver_images")
    .select("png", "label")
    .where("label IS NOT NULL")
    .limit(2048)
    .collect()
)
classes = sorted({r["label"] for r in rows})
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
start = time.perf_counter()
images = np.stack(
    [
        (np.asarray(Image.open(io.BytesIO(r["png"])).convert("RGB"), np.float32) / 255 - mean) / std
        for r in rows
    ]
).transpose(0, 3, 1, 2)
result["decode_images_per_s"] = round(len(rows) / (time.perf_counter() - start))
labels = torch.tensor([classes.index(r["label"]) for r in rows])
real_x = torch.from_numpy(images)
torch.use_deterministic_algorithms(False)
torch.manual_seed(0)
model = build_model(pretrained=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
loss_fn = nn.CrossEntropyLoss()
start = time.perf_counter()
for _epoch in range(2):
    for i in range(0, len(rows), 64):
        xb, yb = real_x[i : i + 64].to(device, non_blocking=True), labels[i : i + 64].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
torch.cuda.synchronize() if device.type == "cuda" else None
result["real_train_images_per_s"] = round(2 * len(rows) / (time.perf_counter() - start))
result["real_images"] = len(rows)
print("GPU_CHECK_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
