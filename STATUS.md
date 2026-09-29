# Status

Updated 2026-09-29. Phase 0 (fix the science) has results and is **awaiting your sign-off**.
Phase 1 has not started.

## Done

- 3-seed run on CPU finished 2026-09-28 06:58 (code `2ccda00`). Results committed in
  `results/phase0/`; README results table filled from them.
- CIFAR-10 Mahalanobis investigation: seed 1 retrained with checkpoint saving (code
  `06245f4`, `results/phase0_seed1_rerun/`), checks on pretrained and fine-tuned features
  (`results/phase0/checks/`), written up in `docs/phase0-note.md`.
- All seven Phase 0 checkboxes in PLAN.md ticked.
- silent-failures #5 (no checkpoint saved) and #6 (a fixed seed did not reproduce a run).

## Open decision before Phase 0 can close

PLAN.md's "Done when" says one command **reproduces every number**. It does not, bit for
bit: a rerun of seed 1 gave balanced accuracy 81.0% instead of 76.5%. Options:

1. Reword to "reproduces every number within the reported run-to-run variation", and
   report rerun spread alongside seed spread.
2. Make CPU training deterministic (fixed thread count, deterministic algorithms), verify
   that two short runs match exactly, then rerun all 3 seeds (about 12 hours).

## Local files (outside the repo)

- Data: `C:\Users\Erjon\data\cdm` (HAM10000, PathMNIST 224, CIFAR-10).
- Weights: `C:\Users\Erjon\data\cdm\checkpoints\phase0_seed1_rerun\seed1.pt`.
- Cached features: `C:\Users\Erjon\data\cdm\features\`.
- Run logs: `logs\` (gitignored).
