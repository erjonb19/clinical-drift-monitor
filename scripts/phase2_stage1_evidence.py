"""Export Phase 2 build step 1 evidence: client splits, the site label rule, and the gates.

Usage: python scripts/phase2_stage1_evidence.py --run <job run id>
Writes results/phase2/stage1_silver.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, job_run, records  # noqa: E402

OUT = Path("results/phase2/stage1_silver.json")
C = "workspace.cdm"
SCC = "Squamous cell carcinoma, NOS"
QUERIES = {
    "gate_results": f"SELECT name, violations, detail FROM {C}.gate_results ORDER BY name",
    "clients_by_split": f"""
SELECT role, client, split, count(*) AS images, count(DISTINCT lesion_id) AS lesions
FROM {C}.silver_metadata GROUP BY ALL ORDER BY role, client, split""",
    "akiec_before_and_after_rule": f"""
SELECT source, role,
       count_if(isic_label = 'akiec') AS akiec_by_isic_mapping,
       count_if(label = 'akiec') AS akiec_after_rule,
       count_if(diagnosis_3 = '{SCC}') AS scc_nos_images,
       count_if(diagnosis_3 = '{SCC}' AND label IS NULL) AS scc_nos_unlabelled
FROM {C}.silver_metadata GROUP BY source, role ORDER BY role, source""",
    "labelled_by_client_and_class": f"""
SELECT client, label, count(*) AS images
FROM {C}.silver_metadata WHERE role = 'client' AND split IN ('train', 'val', 'test')
GROUP BY ALL ORDER BY client, label""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    sql = Sql()
    evidence: dict[str, Any] = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": job_run(args.run),
        **{name: records(sql(query)) for name, query in QUERIES.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
