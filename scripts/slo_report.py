"""Check the model targets (cdm.slo, config/slo.json) on the committed replay months.

Writes results/phase3/monthly/slo_replay.json: for each of the 12 replay months, every target's
status and whether the month would have alerted. The replay is real images in an order we
chose (results/phase3/monthly/spec.json), not drift observed over time. The committed month
records predate the melanoma-referral count, so that target reads "insufficient data" for every
month here, unless results/phase3/monthly/melanoma_backfill.json (written by the Databricks
backfill, databricks/notebooks/melanoma_backfill.py, and committed) supplies the counts. Month m
then sees the backfilled records of months before m only, as its run would have.

Usage: python scripts/slo_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cdm.backfill import merge_history
from cdm.slo import alerts, evaluate, load_config

REPO = Path(__file__).resolve().parents[1]
MONTHLY = REPO / "results" / "phase3" / "monthly"
OUT = MONTHLY / "slo_replay.json"
BACKFILL = MONTHLY / "melanoma_backfill.json"


def main() -> None:
    config = load_config(REPO / "config" / "slo.json")
    records = [json.loads((MONTHLY / f"month_{m:02d}.json").read_text()) for m in range(1, 13)]
    backfill = json.loads(BACKFILL.read_text())["months"] if BACKFILL.exists() else []
    months, labelled = [], []
    for record in records:
        if record.get("delayed_accuracy_of_month"):
            labelled.append(record["delayed_accuracy_of_month"])
        history = merge_history(labelled, backfill, before_month=int(record["month"]))
        slo = evaluate(record, history, config)
        months.append(
            {
                "month": record["month"],
                "by_source": record["by_source"],
                "slo": slo,
                "alerts": alerts(record, slo),
            }
        )
    OUT.write_text(
        json.dumps(
            {
                "what": "model targets checked on the committed 12-month replay "
                "(real images, order chosen by us)",
                "built_by": "scripts/slo_report.py",
                "targets": "config/slo.json",
                "melanoma_backfill": BACKFILL.name if BACKFILL.exists() else None,
                "months": months,
            },
            indent=2,
        )
        + "\n"
    )
    for m in months:
        print(m["month"], {k: v["status"] for k, v in m["slo"].items()}, m["alerts"])


if __name__ == "__main__":
    main()
