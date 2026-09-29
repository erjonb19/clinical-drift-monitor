"""Phase 1 gold task: embeddings, per-image scores, baseline and feature statistics.

Runs as a Databricks job task (entry point ``cdm-gold``) after the pipeline has built
silver. Phase 1 embeds with the pretrained ImageNet EfficientNet-B0 (model version
``imagenet-effb0``), because no registered model exists yet; Phase 2 recomputes these tables
with the registered model. Mahalanobis statistics are fit on HAM10000 train embeddings only,
with PCA 256 as in Phase 0.

Writes ``gold_embeddings``, ``gold_scores``, ``gold_baseline`` and ``gold_feature_stats``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0

from cdm.data import IMAGENET_MEAN, IMAGENET_STD
from cdm.ood import MahalanobisDetector
from cdm.train import features_and_logits

MODEL_VERSION = "imagenet-effb0"
# torchvision names the file after the first 8 hex digits of its SHA-256.
WEIGHTS_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def load_imagenet(weights: Path) -> nn.Module:
    """ImageNet EfficientNet-B0 from a local file, refused unless its hash matches the name."""
    expected = weights.stem.rsplit("-", 1)[-1]
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if not actual.startswith(expected):
        raise ValueError(f"{weights}: SHA-256 {actual[:8]}..., expected {expected}...")
    model = cast(nn.Module, efficientnet_b0(weights=None))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def _tensor(png: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(png)) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1).copy())


@torch.no_grad()
def embed(pngs: Iterable[bytes], model: nn.Module, batch_size: int = 64) -> Iterator[np.ndarray]:
    """1,280-d pooled features for each 224 x 224 PNG, yielded one batch at a time."""
    batch: list[torch.Tensor] = []
    for png in pngs:
        batch.append(_tensor(png))
        if len(batch) == batch_size:
            yield features_and_logits(model, torch.stack(batch))[0].numpy()
            batch = []
    if batch:
        yield features_and_logits(model, torch.stack(batch))[0].numpy()


def score(emb: pd.DataFrame, n_components: int = 256) -> pd.DataFrame:
    """Mahalanobis score for every image, fit on HAM10000 train embeddings only.

    ``emb`` has ``source``, ``split``, ``label`` and ``features`` (one 1,280-d array per row).
    """
    train = emb[(emb["source"] == "ham10000") & (emb["split"] == "train")]
    labels = sorted(train["label"].unique())
    det = MahalanobisDetector(n_components=n_components).fit(
        np.stack(train["features"].to_list()).astype(np.float64),
        train["label"].map({c: i for i, c in enumerate(labels)}).to_numpy(np.int64),
    )
    out = emb.drop(columns="features").copy()
    out["mahalanobis"] = det.score(np.stack(emb["features"].to_list()).astype(np.float64))
    out["model_version"] = MODEL_VERSION
    return out


def baseline(scores: pd.DataFrame) -> pd.DataFrame:
    """The score distribution the drift job compares new batches against: HAM10000's
    validation and test splits, never train (the detector was fit on it)."""
    held_out = scores[(scores["source"] == "ham10000") & scores["split"].isin(["val", "test"])]
    rows = [
        {
            "split": split,
            "n": len(g),
            **{f"q{int(q * 100):02d}": float(g["mahalanobis"].quantile(q)) for q in QUANTILES},
        }
        for split, g in held_out.groupby("split")
    ]
    return pd.DataFrame(rows).assign(model_version=MODEL_VERSION)


def feature_stats(scores: pd.DataFrame, emb: pd.DataFrame) -> pd.DataFrame:
    """Per source and per source x skin type: image count, mean feature norm, score quantiles."""
    joined = scores.assign(feature_norm=[float(np.linalg.norm(f)) for f in emb["features"]])
    joined["skin_type"] = joined["fitzpatrick_skin_type"].fillna("not recorded")
    rows: list[dict[str, Any]] = []
    for keys, group in [
        *(((s, "all"), g) for s, g in joined.groupby("source")),
        *joined.groupby(["source", "skin_type"]),
    ]:
        source, skin = cast(tuple[str, str], keys)
        rows.append(
            {
                "source": source,
                "skin_type": skin,
                "n": len(group),
                "mean_feature_norm": float(group["feature_norm"].mean()),
                "mahalanobis_median": float(group["mahalanobis"].median()),
                "mahalanobis_q95": float(group["mahalanobis"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).assign(model_version=MODEL_VERSION)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs Databricks
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    spark = SparkSession.builder.getOrCreate()
    table = f"{args.catalog}.{args.schema}"
    cols = ["isic_id", "source", "split", "label", "fitzpatrick_skin_type", "png"]
    silver = spark.read.table(f"{table}.silver_images").select(*cols).orderBy("source", "isic_id")
    meta = silver.drop("png").toPandas()
    model = load_imagenet(args.weights)
    pngs = (row["png"] for row in silver.select("png").toLocalIterator())
    features = np.concatenate(list(embed(pngs, model, args.batch_size)))
    emb = meta.assign(features=list(features.astype(np.float32)))
    print(f"embedded {len(emb)} images", flush=True)

    scores = score(emb)
    outputs = {
        "gold_embeddings": meta.assign(
            model_version=MODEL_VERSION, features=[f.tolist() for f in emb["features"]]
        ),
        "gold_scores": scores,
        "gold_baseline": baseline(scores),
        "gold_feature_stats": feature_stats(scores, emb),
    }
    for name, frame in outputs.items():
        (
            spark.createDataFrame(frame)
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{table}.{name}")
        )
        print(f"wrote {table}.{name}: {len(frame)} rows", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
