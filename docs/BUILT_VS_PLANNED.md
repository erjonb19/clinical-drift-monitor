# Built vs planned

What is built, what is scaffolded, and what has not been started. This file is the
reference for what may be claimed about the project: nothing is listed as built until it
runs and is reproducible from the repository.

**This is a personal portfolio project.** No patients, clinicians or business decisions
depend on it.

Last updated: 2026-09-29.

## Built

Working, tested, and run on real data. Results in `results/phase0/`.

### Phase 0: fix the science (done, signed off 2026-09-29)

- HAM10000 download from Harvard Dataverse with MD5 checks and metadata validation;
  PathMNIST (224 px) and CIFAR-10 as fixed 2,000-image OOD subsets (`src/cdm/data.py`).
- Lesion-level 70/15/15 split, stratified by diagnosis: 5,229 / 1,120 / 1,121 lesions.
  Tested for leakage on synthetic data in CI and on the real metadata locally.
- EfficientNet-B0 fine-tuning with class-weighted loss, best epoch chosen on validation
  balanced accuracy (`src/cdm/train.py`). Best weights saved beside the data.
- Three OOD detectors, all fit on training data only and scored the same way for every
  set: Mahalanobis (nearest class mean, PCA 256), max softmax probability, energy
  (`src/cdm/ood.py`). AUROC, FPR@95TPR, mean and range over seeds (`src/cdm/eval.py`).
- One command (`python -m cdm.reproduce`) that saves each seed as it finishes, resumes
  from saved seeds, and refuses an AUROC below 50% or uncommitted code.
- Deterministic CPU training from code `0427b47`: seeded before the model is built, fixed
  thread count, PyTorch deterministic algorithms, seeded data workers. Tested in CI and
  checked once on real data.
- **Measured:** balanced accuracy 78.0% (76.4–79.3) over 3 seeds on CPU. OOD results
  and their caveats in the README.

Known limits, documented rather than fixed:
- The reported seeds came from code that seeded after building the model, so their
  ranges understate run-to-run variation; the measured rerun spread is reported next to
  them ([silent-failures #6](silent-failures.md)). Not rerun, by decision.
- Mahalanobis on CIFAR-10 is unstable and sensitive to PCA size
  ([phase0-note.md](phase0-note.md)).

## Scaffolded

- CI: ruff, ruff format, mypy (strict) and pytest on every push. No eval gate yet
  (Phase 3).

## Not started

- Phases 1 to 6 as listed in `PLAN.md`. The course notebook in `reference/` is history,
  not a starting point; its numbers are superseded by Phase 0.
