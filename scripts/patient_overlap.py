"""Patient-level overlap between splits, on the current silver table.

Splits are by lesion (Phase 0's rule, frozen since v1). ISIC records ``patient_id`` for some
collections, so a patient with several lesions can have one in train and another in test. This
script counts that per source from ``workspace.cdm.silver_metadata`` and writes:

- results/phase2/patient_overlap/patient_overlap.json: per-source counts, and the test lesions
  whose patient also has a train lesion (used for the sensitivity rows from Phase B on);
- config/patient_overlap_v1.csv: the documented cross-split patients, which the
  ``patients_in_one_split`` gate allows; any other patient fails it.

Sources without ``patient_id`` (HAM10000, Barcelona, part of MSK) cannot be checked; the JSON
says so per source rather than reporting zero.

With ``--predictions`` (the analysis look's committed per-image probabilities), it also reports
how much the overlap moves the test numbers: each model's test metrics with and without the test
lesions that share a patient with train, and the ensemble's 95% bootstrap intervals resampled by
patient where ISIC records one and by lesion elsewhere. Saved predictions only: no new look.

Usage: python scripts/patient_overlap.py
           [--predictions results/phase2/analysis_look/predictions.csv.gz]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql  # noqa: E402

from cdm.eval import classification_metrics
from cdm.final import bootstrap_by_lesion
from cdm.gates import patients_across_splits
from cdm.splits import CLASSES, SPLITS

OUT = Path("results/phase2/patient_overlap")
CSV = Path("config/patient_overlap_v1.csv")
QUERY = """SELECT source, isic_id, lesion_id, patient_id, split, role, label
FROM workspace.cdm.silver_metadata"""


def per_source(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    crossing = patients_across_splits(frame)
    for source, g in frame.groupby("source"):
        known = g[g["patient_id"].notna()]
        lesions_per_patient = known.groupby("patient_id")["lesion_id"].nunique()
        row: dict[str, Any] = {
            "role": str(g["role"].iloc[0]),
            "images": len(g),
            "lesions": int(g["lesion_id"].nunique()),
            "images_with_patient_id": len(known),
            "patients": int(known["patient_id"].nunique()),
            "patients_with_more_than_one_lesion": int((lesions_per_patient > 1).sum()),
        }
        if not len(known):
            row["checkable"] = False
            out[str(source)] = row
            continue
        row["checkable"] = len(known) == len(g) or "partly: only images with patient_id"
        trained = known[known["split"].isin(SPLITS)]
        if len(trained):
            cross = {p for s, p in crossing if s == source}
            train_patients = set(trained.loc[trained["split"] == "train", "patient_id"])
            test = trained[trained["split"] == "test"]
            shared = test[test["patient_id"].isin(train_patients)]
            row.update(
                patients_in_split_rows=int(trained["patient_id"].nunique()),
                patients_in_more_than_one_split=len(cross),
                lesions_of_those_patients=int(
                    trained.loc[trained["patient_id"].isin(cross), "lesion_id"].nunique()
                ),
                test_lesions=int(test["lesion_id"].nunique()),
                test_lesions_sharing_a_patient_with_train=int(shared["lesion_id"].nunique()),
                test_images_sharing_a_patient_with_train=len(shared),
                test_lesion_ids_sharing_a_patient_with_train=sorted(
                    map(str, shared["lesion_id"].unique())
                ),
            )
        out[str(source)] = row
    return out


METRICS = ("balanced_accuracy", "melanoma_sensitivity", "macro_auroc")
ENSEMBLE = "v2-ensemble-final-seeds0-1-2"


def _metrics(rows: pd.DataFrame, model: str) -> dict[str, Any]:
    probs = rows[[f"{model}__{c}" for c in CLASSES]].to_numpy(dtype=np.float64)
    logits = np.log(np.clip(probs, 1e-12, None))
    labels = rows["label"].map(CLASSES.index).to_numpy(dtype=np.int64)
    m = classification_metrics(labels, logits)
    out: dict[str, Any] = {"images": len(rows), **{k: m[k] for k in METRICS}}
    if model == ENSEMBLE:
        # Cluster by patient where ISIC records one, else by lesion.
        clusters = rows["patient_id"].fillna(rows["lesion"]).astype(str).tolist()
        out["bootstrap_by_patient_or_lesion"] = bootstrap_by_lesion(logits, labels, clusters)
    return out


def sensitivity(frame: pd.DataFrame, predictions: Path, shared: set[str]) -> dict[str, Any]:
    """Client-test metrics with and without the test lesions whose patient has a train lesion."""
    preds = pd.read_csv(predictions)
    test = preds[(preds["split"] == "test") & preds["label"].notna()].merge(
        frame[["isic_id", "patient_id"]], on="isic_id", how="left"
    )
    models = sorted({c.split("__")[0] for c in preds.columns if "__" in c})
    out: dict[str, Any] = {"predictions": predictions.as_posix(), "views": {}}
    views = {"pooled_client_test": test, "msk_test": test[test["source"] == "msk"]}
    for view, rows in views.items():
        kept = rows[~rows["lesion"].astype(str).isin(shared)]
        out["views"][view] = {
            model: {"all": _metrics(rows, model), "without_shared_patients": _metrics(kept, model)}
            for model in models
        }
    return out


def main() -> None:
    # Subsets without df or vasc make sklearn warn on every bootstrap draw; the metrics are fine.
    warnings.filterwarnings("ignore", "y_pred contains classes not in y_true")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    frame = Sql()(QUERY)
    crossing = patients_across_splits(frame)
    result: dict[str, Any] = {
        "computed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "table": "workspace.cdm.silver_metadata",
        "rule": "splits are by lesion; a patient is counted once per source by ISIC patient_id",
        "silver_images": len(frame),
        "patients_in_more_than_one_split": len(crossing),
        "by_source": per_source(frame),
    }
    if args.predictions:
        shared = set(result["by_source"]["msk"]["test_lesion_ids_sharing_a_patient_with_train"])
        result["test_sensitivity"] = sensitivity(frame, args.predictions, shared)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patient_overlap.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    with CSV.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["source", "patient_id", "splits"])
        for (source, patient), splits in sorted(crossing.items()):
            writer.writerow([source, patient, "|".join(splits)])
    by_source: dict[str, dict[str, Any]] = result["by_source"]
    summary = {
        s: {k: v for k, v in r.items() if not k.startswith("test_lesion_ids")}
        for s, r in by_source.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
