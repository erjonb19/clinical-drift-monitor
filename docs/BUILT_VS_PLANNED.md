# Built vs planned

What is built, what is scaffolded, and what has not been started. This file is the
reference for what may be claimed about the project: nothing is listed as built until it
runs and is reproducible from the repository.

**This is a personal portfolio project.** No patients, clinicians or business decisions
depend on it.

Last updated: 2026-10-20.

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

### Phase 1: data engineering (built 2026-09-29)

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
- Auto Loader's incremental pickup of new files: exercised on 2026-10-04 by the v2 data
  update (job run 708528812450324). bronze_images grew by exactly the 4,999 new Barcelona
  files (22,657 to 27,656), confirmed by the update's flow metrics; every gate passed,
  including frozen_splits_kept. Evidence: `results/phase2/v2/data_checks.json`.
- Databricks reports only the first failing gate; the rest are confirmed by running
  `cdm.gates.run_all` on silver, as the note describes.
- Gold embeddings use pretrained ImageNet features; Phase 2 recomputes them.

### Phase 2: ML (built 2026-09-29 to 2026-10-06)

Run on Databricks serverless GPU (1x A10G) and serverless CPU; training runs in MLflow
(experiment `cdm-phase2`). Every committed result is also an MLflow run in `cdm-evidence`
(`results/mlflow_runs.json`; Phase 0 to 2 files were logged after the fact on 2026-10-20 and are
tagged so). Record: [phase2-note.md](phase2-note.md).

- Silver splits for Barcelona and MSK by lesion, with HAM10000's Phase 0 split pinned and v1
  lesions frozen in v2 (`config/splits_v1.csv`, gate `frozen_splits_kept`).
- v2 data: Barcelona to 9,999 images. Auto Loader picked up only the new files.
- Tuning ladder on validation only: no rung beat the control (`results/phase2/ladder/`).
- 3 centralized and 6 federated models (Flower FedAvg and FedProx, clients trained in sequence in
  one process). One test look per model (`results/phase2/v2/test/final_test.json`).
- Drift table: 4 detectors × 3 setups × 2 real held-out sites + 2 benchmark sets
  (`results/phase2/drift/v2/`). Mahalanobis was shipped by the pre-set rule.
- Triage operating point, committed before test (`results/phase2/triage/`).
- Label provenance, the Buenos Aires local recalibration, and the analysis look (seven-class
  external accuracy, results by skin type, worst misses; `results/phase2/analysis_look/`).
- Patient-level split check, run on silver (`results/phase2/patient_overlap/`, silent-failures #9).
  The `patients_in_one_split` gate is part of `gate_results`, which fails a pipeline update on
  any violation. It first ran in research Phase A's updates (`61686b98`, `5992b755`), and both
  completed. Its row from those updates is not committed.

Known limits:
- Splits are by lesion. 47 MSK patients span splits, and HAM10000 and Barcelona cannot be
  checked. The measured test effect is small.
- Detector scores do not reproduce between GPU and CPU, so batch-level drift and the gold
  recompute moved to Phase 3.
- Error analysis is partly done: the confusion matrices by site and skin type and the review
  worksheet are research Phase D.
- Nothing is registered in Unity Catalog yet (Phase 3).

### Research Phase A: OMOP tables (built 2026-10-21)

- `config/omop_concepts.csv`: 107 source values mapped to 67 OMOP concepts, each checked
  against two Athena downloads (`scripts/omop_concepts.py`). 15 values have no verified
  concept (0, with reasons), including modality: no DICOM vocabulary was offered.
- `cdm.omop`: person, observation_period, condition_occurrence, procedure_occurrence,
  image_occurrence, cdm_source and concept, plus seven gates. The pipeline publishes them as
  `workspace.cdm.omop_*`, with `omop_gate_results` (fails the update) and `omop_profile`.
  The first build (update `61686b98`, code `cf10fee`) passed every gate. It was rebuilt
  with Gender concepts (update `5992b755`, code `0239a19`), which also passed every gate.
- `docs/data_dictionary.md` is generated from the column definitions and the build, and a
  test keeps it current.
- Unity Catalog lineage from bronze through silver into every OMOP table, read from
  `system.access.table_lineage` the next day (the table lags):
  `results/research/a_omop/lineage.json`. A test checks every OMOP table has a silver edge.
- Not built: the `observation` table listed in PLAN.md. Every mapped field has a home in
  another table, so it would be empty.

### Research Phase G: research informatics services (2026-10-21 and 22)

- G1: `docs/EVAL_PLAN.md`, `docs/governance.md`, the request issue form and the project board.
- G2: the governed cohort service and phenotype library, with tests; run on the workspace OMOP
  tables for seven synthetic requests (#8 to #14). Stand-in decisions are in
  `results/research/g2_cohort/synthetic_approvals.jsonl`; `phenotypes/approvals.jsonl` is for
  the project owner's own decisions and is empty.
- G3: dataset metadata, `CITATION.cff` and the quickstart notebook, run in CI offline. The
  Zenodo release is not made: it needs the owner to connect Zenodo.
- G4: `scripts/service_metrics.py` and `docs/service_metrics.md`, with tests; request numbers
  come from a committed snapshot of the issue tracker.

### Tooling

- `scripts/log_evidence.py`: committed results file to MLflow run, refusing uncommitted files.
  CI checks every indexed file still matches its logged hash.

## Scaffolded

### Research Phase G5: clinical text extraction

- `cdm.extract` and `scripts/extraction.py`: note sources (MTSamples; MIMIC-IV-Note behind a
  tier 2 switch, never sent to an outside API), label choices from the concept map, the Haiku
  extractor with a spend log and budget cap, a keyword baseline, the labelling sheet, scoring
  and error categories. Tested with made-up notes and a recorded transport. One live call on a
  made-up note checked the connection. Not run on any real note: no MTSamples file and no
  labels yet, so nothing is scored or logged to MLflow.

### Tooling

- CI: ruff, ruff format, mypy (strict) and pytest on every push. No eval gate yet
  (Phase 3).

## Not started

- Research extension Phases B to F ([research_extension_plan.md](research_extension_plan.md)), and
  G6 (the Streamlit demo).
- Phases 3 to 6 as listed in `PLAN.md`. The course notebook in `reference/` is history,
  not a starting point; its numbers are superseded by Phase 0.
