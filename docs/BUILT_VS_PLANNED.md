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

### Phase 1: data engineering (built 2026-09-29, awaiting sign-off)

Run on Databricks Free Edition; evidence in `results/phase1/run_evidence.json`, explained in
[phase1-note.md](phase1-note.md).

- Ingest job task (`src/cdm/ingest.py`): lands 22,657 images (HAM10000 10,015 from ISIC
  collection 212; Barcelona 5,000, capped and pinned; Buenos Aires 1,616; MSK 3,728;
  PAD-UFES-20 2,298) with a manifest and SHA-256 per file. Reruns download nothing already
  landed and reuse stored hashes.
- Declarative pipeline (`pipelines/lakehouse.py`): Auto Loader bronze, silver with Phase 0's
  split reproduced exactly (`cc2b196cd5bf58ad`), a quarantine table, and 11 gates that fail
  the update. All 11 passed on real data.
- Gold job task (`src/cdm/gold.py`): embeddings (model version `imagenet-effb0`), per-image
  scores, the val/test baseline, and per-site and per-skin-type statistics.
- Broken-input demo: a small faulted copy fails the update on `gate_results`, once (no
  retries), with exactly the six expected gates failing.
- Deployment by CLI (`scripts/deploy_databricks.py`, `databricks/*.json`) and the runbook.

Known limits:
- Auto Loader's incremental pickup of new files has not been exercised: the one successful
  update read every file. Phase 3's monthly job is its first real test.
- Databricks reports only the first failing gate; the rest are confirmed by running
  `cdm.gates.run_all` on silver, as the note describes.
- Gold embeddings use pretrained ImageNet features; Phase 2 recomputes them.

## Scaffolded

### Tooling

- CI: ruff, ruff format, mypy (strict) and pytest on every push. No eval gate yet
  (Phase 3).

## Not started

- Phases 2 to 6 as listed in `PLAN.md`. The course notebook in `reference/` is history,
  not a starting point; its numbers are superseded by Phase 0.
