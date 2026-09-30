# Status

Updated 2026-09-29. Phase 0 is done (signed off 2026-09-29). **Phase 1 (data engineering) is
built on Databricks and awaiting your sign-off.** Phase 2 has not started.

## Phase 1 outcome

- "Done when" demonstrated: job run `606896740511548` rebuilt bronze to gold with all 11
  gates passing on 22,657 real images, and the broken-input demo (run `161990905473833`)
  failed its update once on `gate_results` with exactly the six expected gates failing.
- All five Phase 1 checkboxes in PLAN.md ticked, each with a "(Built: …)" note where the
  build differs from the wording (Auto Loader's incremental pickup not yet exercised;
  duplicate files fail a gate; gold runs as a job task after the pipeline).
- Evidence: `results/phase1/run_evidence.json`. Explanation: `docs/phase1-note.md`.

## Deployed in your workspace

- Jobs: `cdm-phase1` (143178896044655), `cdm-phase1-broken-demo` (1119799696513742).
- Pipelines: `cdm-phase1`, `cdm-phase1-broken`; both stop on the first failed update.
- Tables in `workspace.cdm` (and the demo's in `workspace.cdm_broken`); files in the volume
  `workspace.cdm.raw`. How to rerun: `docs/phase1-runbook.md`.

## Waiting on you

- Sign-off on Phase 1.
- Before Phase 2: its plan, for your OK (CLAUDE.md).
- Still open in PLAN.md: checks B (bundles) and C (data quality monitoring); not needed for
  the CLI route.

## Local files (outside the repo)

- Data: `C:\Users\Erjon\data\cdm` (HAM10000, PathMNIST 224, CIFAR-10).
- Weights and checkpoints: `C:\Users\Erjon\data\cdm\checkpoints\`,
  `C:\Users\Erjon\.cache\torch\hub\checkpoints\`.
- Run logs: `logs\` (gitignored). Built wheel and rendered definitions: `dist\` (gitignored).
