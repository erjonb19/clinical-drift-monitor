# Status

Updated 2026-09-29. **Phase 0 (fix the science) is done**, signed off 2026-09-29.
Phase 1 (data engineering) has not started: its plan is waiting for your OK.

## Phase 0 outcome

- Results: `results/phase0/` (3 seeds, code `2ccda00`), summarised in the README with
  three caveats: CIFAR-10 Mahalanobis instability, PCA-size sensitivity, and run-to-run
  variation larger than the seed ranges.
- Why the notebook's 4.1% and 43.4% were wrong, and the CIFAR-10 investigation:
  `docs/phase0-note.md`.
- "Done when" reworded to "within the reported run-to-run variation"; the reported seeds
  were not rerun.
- Training is deterministic from code `0427b47` on (test in CI, checked on real data).

## Waiting on you

- Results of the week 1 Databricks checks (ISIC S3 access, bundles, data quality
  monitoring) from your Free Edition workspace. Phase 1's design depends on them.
- OK on the Phase 1 plan.

## Local files (outside the repo)

- Data: `C:\Users\Erjon\data\cdm` (HAM10000, PathMNIST 224, CIFAR-10).
- Weights: `C:\Users\Erjon\data\cdm\checkpoints\` (seed 1 rerun, smoke runs).
- Cached features: `C:\Users\Erjon\data\cdm\features\`.
- Run logs: `logs\` (gitignored).
