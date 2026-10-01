"""Phase 2 scoring report, shared by the training and evaluation notebooks.

Scores rows that have ``client``, ``split``, ``label``, ``lesion_id`` and ``bordered`` (dark
image corners, Barcelona's vignette), aligned with model logits. Groups: pooled, HAM10000
over both institutions, each client, and Barcelona split by bordered and non-bordered images.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cdm.eval import classification_metrics, confusion
from cdm.splits import CLASSES

SMALL_CLASS = 20  # flag classes with fewer test images than this


def score_rows(
    frame: pd.DataFrame, logits: NDArray[np.float64], labels: NDArray[np.int64]
) -> dict[str, dict[str, Any]]:
    """Metrics and a confusion matrix per split and group. ``frame`` row i matches
    ``logits[i]`` and ``labels[i]``. Empty groups are left out."""
    out: dict[str, dict[str, Any]] = {}
    for split in sorted(frame["split"].unique()):
        in_split = (frame["split"] == split).to_numpy()
        groups = {
            "pooled": in_split,
            "ham10000_both_institutions": in_split
            & frame["client"].str.startswith("ham_").to_numpy(),
        }
        for client in sorted(frame["client"].unique()):
            groups[client] = in_split & (frame["client"] == client).to_numpy()
        is_bcn = (frame["client"] == "barcelona").to_numpy()
        bordered = frame["bordered"].to_numpy(dtype=bool)
        groups["barcelona_bordered"] = in_split & is_bcn & bordered
        groups["barcelona_not_bordered"] = in_split & is_bcn & ~bordered
        out[split] = {
            name: {**classification_metrics(labels[mask], logits[mask]),
                   "confusion": confusion(labels[mask], logits[mask]), "images": int(mask.sum())}
            for name, mask in groups.items()
            if mask.any()
        }  # fmt: skip
    return out


def support(frame: pd.DataFrame) -> dict[str, Any]:
    """Images and lesions per split, client and class, and the test classes under
    SMALL_CLASS images (pooled and per client). Counts only: no model output."""
    counts: dict[str, Any] = {}
    for (split, client), g in frame.groupby(["split", "client"]):
        counts[f"{split}/{client}"] = {
            c: {"images": int((g["label"] == c).sum()),
                "lesions": int(g.loc[g["label"] == c, "lesion_id"].nunique())}
            for c in CLASSES
        }  # fmt: skip
    test = frame[frame["split"] == "test"]
    small = {
        "pooled": [c for c in CLASSES if (test["label"] == c).sum() < SMALL_CLASS],
        **{
            client: [c for c in CLASSES if (g["label"] == c).sum() < SMALL_CLASS]
            for client, g in test.groupby("client")
        },
    }
    return {"by_split_client_class": counts, f"test_classes_under_{SMALL_CLASS}_images": small}
