# Status

Updated 2026-09-29. Phase 0 is done (signed off 2026-09-29). **Phase 1 (data engineering)
code is written and tested locally; it has not run on Databricks yet.**

## Phase 1: done

- PLAN.md updated with the approved changes: sites defined by collection, `imagenet-effb0`
  embeddings in Phase 1, lesion splits wherever data trains, skin type where recorded, the
  5,000-image Barcelona cap, HAM10000 from ISIC collection 212, the /tmp-then-copy route,
  and CLI deployment.
- Source checks against the live ISIC API (`results/phase1/isic_checks.json`): the
  diagnosis mapping reproduces all 10,015 HAM10000 labels; ISIC's HAM10000 files are
  re-encoded (not byte-identical to Phase 0's); Barcelona's selection is pinned
  (fingerprint `0f156bf4e99bb8c2`).
- Ingest task, pipeline, gold task, gates, deployment and runbook, with tests (67 pass).

## Next: your steps (docs/phase1-runbook.md)

1. Install and log in to the Databricks CLI (runbook steps 1 and 2).
2. Deploy: `scripts\deploy_databricks.py --dry-run`, then without `--dry-run` (step 3).
3. Run job `cdm-phase1` (step 4); expect about 26,600 images on the first run.
4. Run job `cdm-phase1-broken-demo` (step 5); it must fail on `gate_results`.
5. Report back: task statuses, `gate_results`, row counts per source, and any error text.

The first Databricks run is the first test of the pipeline file, CLI commands and job
definitions, so expect to report a first-run error or two for Claude to fix.

## Still open

- Checks B (bundles) and C (data quality monitoring) in your workspace; not needed for the
  CLI route, but PLAN.md's open questions ask for them.

## Local files (outside the repo)

- Data: `C:\Users\Erjon\data\cdm` (HAM10000, PathMNIST 224, CIFAR-10).
- Weights and checkpoints: `C:\Users\Erjon\data\cdm\checkpoints\`,
  `C:\Users\Erjon\.cache\torch\hub\checkpoints\`.
- Run logs: `logs\` (gitignored). Built wheel and rendered definitions: `dist\` (gitignored).
