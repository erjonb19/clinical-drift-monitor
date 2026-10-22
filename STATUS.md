# Status

Updated 2026-10-20.

- **Phases 0 to 2 are built.** Phase 2's last open item, error analysis, moved to research
  Phase D. The research-extension prep is on branch `research/0-prep`:
  - patient-level split check and gate;
  - analysis look results exported;
  - MLflow evidence logging;
  - PLAN.md amendment and docs refresh.
- **Next:** research Phase A (OMOP and SNOMED CT gold tables), once the prep PR is merged.
- **Phases 3 to 6** have not started.

## Deployed in your workspace

- **Phase 1 jobs:** `cdm-phase1` (143178896044655), `cdm-phase1-broken-demo` (1119799696513742).
- **Phase 2 jobs:**
  - `cdm-phase2-overnight` (409896132564474)
  - `cdm-phase2-v2` (403273250846193)
  - `cdm-phase2-v2-test` (778954396717951)
  - `cdm-phase2-drift-table` (144921606000493)
  - `cdm-phase2-triage` (462468285107739)
  - `cdm-phase2-batch-drift-reference` (1057242575787924)
  - `cdm-phase2-ba-recalibration` (296642112747510)
  - `cdm-phase2-analysis-look` (509088198910916)
- **Tables** are in `workspace.cdm`, with the demo's in `workspace.cdm_broken`. Files are in the
  volume `workspace.cdm.raw`.
- **MLflow experiments:** `cdm-phase2` (training) and `cdm-evidence` (one run per committed
  results file).

## Local files (outside the repo)

- **Data:** `C:\Users\Erjon\data\cdm` (HAM10000, PathMNIST 224, CIFAR-10).
- **Weights and checkpoints:** `C:\Users\Erjon\data\cdm\checkpoints\` and
  `C:\Users\Erjon\.cache\torch\hub\checkpoints\`. Phase 2 checkpoints are in the volume.
