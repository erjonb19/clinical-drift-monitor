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

### Research Phase B: BiomedCLIP baseline (built 2026-10-22)

- `cdm.vlm` (pinned BiomedCLIP, zero-shot, drift metrics, paired bootstrap) and `cdm.probe`
  (3-seed linear probe), tested in CI with a fake model, including that drift scores point
  the right way.
- One Databricks serverless GPU job embedded every image (`databricks/vlm_job.json`, run
  1040483728228282, 186 seconds of embedding); embeddings in `workspace.cdm.gold_vlm_embeddings`.
- `scripts/vlm_baseline.py`: `select` on train and validation wrote the committed spec; one
  test look ran as a serverless CPU job (`databricks/vlm_look_job.json`, run
  930314930675108) after a dry run on validation. The lock is written only after the
  results are saved. Results and spec are logged to MLflow; numbers in the README.
- The look's trail: a first attempt on the laptop was stopped by Claude Code for low system
  memory before it wrote its started lock (that code wrote the lock before scoring any test
  image, and no lock file was left), so no metric was computed or seen. Then dry runs on
  validation on the laptop and on Databricks (`results/research/b_vlm/dry_run/`), the
  preflight (`vlm_baseline.py preflight --code-version f4cf563`) passed, and the job ran once
  with `mode=look`. Its wheel (`deploy/f4cf563/cdm-0.1.0-py3-none-any.whl`) had sha256
  `47e1bc6fe2f3ebdc2c5c7b782f16b75f306cd6bdfa459a114c0aee43ba1e4afa` when uploaded; the job
  did not check it (later jobs do).

### Research Phase C: subgroups (built 2026-10-22)

- `cdm.subgroups` (age bands, concept-ID groups, intervals within groups, drift by group, the
  reproduction gate), tested in CI; `databricks/notebooks/subgroups.py` and
  `databricks/subgroups_job.json` (serverless CPU, wheel and inputs hash-checked), staged by
  `scripts/stage_research_job.py`.
- First run (570257730256860) under `spec.json`; after a fresh review, `spec_v2.json` added the
  MSK-overlap row, zero-shot, Fitzpatrick type, site and alert rates by group, and an
  alignment gate for saved drift scores. Second run 534810524836120 (a first v2 attempt failed
  at its first query, before any metric). Both results files and specs are logged to MLflow.
- Known limits: the first run's `results.json` carries intervals for some withheld metrics
  (v2 does not); the notebook in the workspace is imported from the staged commit but its
  source hash is recorded by the staging script, not checked inside the job.

### Research Phase D: Grad-CAM review worksheet (built 2026-10-22)

- `cdm.explain` (Grad-CAM, the weight-randomization check) and `cdm.review` (selection,
  confusion tables), tested in CI; `databricks/notebooks/review_worksheet.py` and its job,
  one run (624234357242388) under the committed spec, with a reproduction gate on the shipped
  checkpoint. Results logged to MLflow.
- The worksheet exists (MLflow artifact, `gold_review_worksheet`, overlays in the volume).
  **Not reviewed:** `reviewer_notes` is blank until a person fills it in.

### Research Phase E: image plus metadata (built 2026-10-22)

- `cdm.fusion` (metadata encoding fit on train, five variants), tested in CI;
  `databricks/notebooks/fusion.py` and its job: selection on validation (run
  230673257459446) committed as the spec, then one test look (run 75228269324996) with an
  attempt log, results saved before the lock. Spec and results logged to MLflow.
- Known limit: the ImageNet-feature variants chose the grid's lowest learning rate; the
  grid was not widened.
- Follow-up (owner's option b): the shipped model's own features, extracted on GPU
  (`databricks/shipped_features_job.json`, run 498023462798649), scored as a new variant
  pair with `cdm.fusion.select_per_seed` and its own spec and single look
  (`databricks/fusion_shipped_job.json`, mode=look committed; run 124072636853443).

### Research Phase G: research informatics services (2026-10-21 and 22)

- G1: `docs/EVAL_PLAN.md`, `docs/governance.md`, the request issue form and the project board.
- G2: the governed cohort service and phenotype library, with tests; run on the workspace OMOP
  tables for seven synthetic requests (#8 to #14). Stand-in decisions are in
  `results/research/g2_cohort/synthetic_approvals.jsonl`; `phenotypes/approvals.jsonl` is for
  the project owner's own decisions and is empty.
- G3: dataset metadata, `CITATION.cff` and the quickstart notebook, run in CI offline. The
  Zenodo release is not made yet: the owner deferred it until after the phenotype approvals.
- G4: `scripts/service_metrics.py` and `docs/service_metrics.md`, with tests; request numbers
  come from a committed snapshot of the issue tracker.

### Tooling

- `scripts/log_evidence.py`: committed results file to MLflow run, refusing uncommitted files.
  CI checks every indexed file still matches its logged hash.

### Phase 3: per-run cost tracking (partly built 2026-10-22)

- `scripts/cost_table.py` builds `workspace.cdm.gold_job_costs` from `system.billing.usage`,
  `system.billing.list_prices` (USD) and `system.lakeflow`: DBUs, GPU or CPU from an
  explicit SKU list, list-price equivalent, wall time and result, per job run and per
  pipeline update. Checks fail the build if the price join drops or duplicates usage rows,
  a row has no price, or a SKU is unknown.
- `results/phase3/costs/result_costs.json` traces each committed results file that names a
  job run to that run's cost. Refreshed 2026-10-23 (billing to 2026-10-22 23:00 UTC): 13 of 31
  traced; the other 18 ran after the billing cutoff. Traced so far: 34.3 GPU DBUs and 26.7 CPU
  DBUs, $31.62 at list price. Free Edition charges nothing. Not scheduled; refreshed by hand.

### Phase 3: serving bundle, risk-coverage and the API (built 2026-10-23; not deployed)

- `cdm.serving` (NumPy Mahalanobis scorer, risk-coverage, the prediction service) and
  `databricks/notebooks/serving_bundle.py`: one CPU run (682399318227929) under the committed
  spec built the bundle in the volume, chose the abstain threshold on validation and made one
  look at test and the new hospitals.
- `app/api/` (FastAPI) and `Dockerfile`: tested in CI and run locally in Docker
  (`scripts/api_local_check.py`, `results/phase3/serving/api_local_checks.json`). Not
  deployed: the Hugging Face Spaces deploy needs the owner's login, and
  `scripts/deploy_space.py` refuses a bundle other than the evaluated one.
- Known gap: the bundle job wrote its attempt log after computing test features (the spec
  said before); stated in the README.

### Phase 3: the monthly drift job (built 2026-10-23)

- `cdm.monitor` (replay schedule, PSI and KS alert rule, per-skin-type alerts, delayed
  accuracy; tested) and `databricks/notebooks/monthly_drift.py`, scheduled monthly. The
  12-month replay ran as one backfill (830837634650904); `gold_drift_monthly`,
  `gold_prediction_audit` and `gold_drift_baseline` exist. No scheduled run yet.

### Phase 3: Unity Catalog model and lineage (partly built 2026-10-23)

- `databricks/notebooks/register_model.py` (run 505700557738191) registered the evaluated serving
  bundle as `workspace.cdm.skin_lesion_classifier` v1 (alias `champion`, bundle hash as a tag), with
  the clients' training split of `silver_images` logged as its input, and read 31 lineage edges from
  `system.access.table_lineage` (`results/phase3/lineage/lineage.json`).
- A second job loaded `models:/workspace.cdm.skin_lesion_classifier@champion` through the registry and
  matched the bundle's answers on made-up images (`results/phase3/lineage/registered_model_check.json`).
  Correction: that file says "fresh environment (pip requirements from the model)"; the load ran in the
  job's own environment, which has the cdm wheel installed, so the model's packaged code and pins were
  not tested in isolation.
- Gap: no lineage edge reaches `gold_prediction_audit`, `gold_drift_monthly` or `gold_drift_baseline`
  (written from pandas);
  the monthly job does not yet load the model through its registry URI.

## Scaffolded

### Phase 3: ONNX export (tried 2026-10-22; parity failed)

- `cdm.serve_onnx` exports each shipped seed with features and logits and a dynamic batch,
  measures parity and latency, and applies the noise-based rule; tested in CI on a toy
  model. `databricks/notebooks/onnx_export.py` measured PyTorch's own batch-size noise
  (run 540132085766008) and then ran the parity check once under the amended spec
  (run 885785618939922): **failed**, ONNX gaps 3 to 7x PyTorch's noise and top-class changes
  on stable images. The root cause is unknown (both the noise and the gap are far above
  usual float32 differences). The ONNX files are in the volume; nothing serves them.

### Research Phase F: prep (2026-10-22)

- F: `docs/cohort_contract.md`, a draft waiting for the owner's review.

### Research Phase G5: clinical text extraction

- `cdm.extract` and `scripts/extraction.py`: note sources (MTSamples; MIMIC-IV-Note behind a
  tier 2 switch, never sent to an outside API), label choices from the concept map, the Haiku
  extractor with a spend log and budget cap, a keyword baseline, the labelling sheet, scoring
  and error categories. Tested with made-up notes and a recorded transport. One live call on a
  made-up note checked the connection. Not run on any real note: no MTSamples file and no
  labels yet, so nothing is scored or logged to MLflow.

### Research Phase G6: public cohort demo

- `app/cohort_demo.py` and `cdm.demo`: the cohort service on public PAD-UFES-20 metadata,
  fetched at start-up; each visitor approves in a scratch file. Its logic is tested in CI on a
  made-up stand-in, and the page ran once locally against live ISIC metadata (Streamlit's
  test harness). Deployed on Streamlit Community Cloud on 2026-10-22, with the owner's approval:
  https://cdm-cohort-demo.streamlit.app/. The live page was opened once: it loaded the live metadata, and Run refused an
  unapproved definition and logged the refusal.

### Tooling

- CI: ruff, ruff format, mypy (strict) and pytest on every push. No eval gate yet
  (Phase 3).

## Not started

- Phases 3 to 6 as listed in `PLAN.md`. The course notebook in `reference/` is history,
  not a starting point; its numbers are superseded by Phase 0.
