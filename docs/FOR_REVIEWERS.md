# For reviewers: evidence map

This maps the duties of clinical research informatics and AI engineering roles to the exact
evidence in this repository. It is a personal portfolio project on public data
(ISIC Archive images, MTSamples notes) and synthetic investigator requests, **not for clinical
use**. Every number named here comes from a committed results file, and each file is also logged
as an MLflow run (`results/mlflow_runs.json`).

## A 2-minute tour

1. **Results first (30 s).** In the README, start with the Phase 2 table. The shipped 3-seed
   EfficientNet-B0 ensemble reaches 66.46% balanced accuracy on the training hospitals' test split.
   At two hospitals it never saw, it falls to 55.91% and 39.40%. That silent degradation is why the
   project has a drift monitor.
2. **How the numbers were protected (30 s).** Open [docs/EVAL_PLAN.md](EVAL_PLAN.md) and any
   `results/**/spec.json`. Every test result follows a specification committed before its single
   test look. Locks and attempt logs record each look, and results are saved before the lock.
   [docs/silent-failures.md](silent-failures.md) lists bugs that reported success while being
   wrong, with the check that now catches each one.
3. **The research questions (30 s).** These are research Phases B to E in the README:
   - Does a medical vision-language model beat it? No: BiomedCLIP's linear probe trails by 15 to
     16 points.
   - Who does it fail? Subgroups by site, skin type, age, sex and body site.
   - Can a reviewer see why? Grad-CAM with a sanity check.
   - Does metadata help? Small gains, partly a shortcut.
4. **It runs as a service (30 s).** This is the README's Phase 3 section:
   - an API with an abstain flag, its Docker image built and smoke-tested in CI (with an
     untrained test bundle; the real bundle was checked locally);
   - a monthly drift job (schedule set; its first scheduled run is due on the 1st of next month)
     whose 12-month replay alerted on every fully shifted month and on none of the
     in-distribution months;
   - a retraining loop whose pre-registered rule declined a challenger that gained 6.2 points
     at the new hospital but lost 1.6 on the original sites;
   - a Unity Catalog model registry, lineage, an audit table and per-run cost tracking.

## Duty by duty

| Duty | Evidence | Result (committed file) |
|---|---|---|
| AI methods benchmarked and validated | Phase 2 training on Databricks GPU (centralized and federated, Flower); research Phase B BiomedCLIP baseline (PR #22); Phase E fusion (PRs #28, #30) | Ensemble 66.46% vs single model 62.99% (seeds 62.21 to 64.28) vs FedAvg 57.14% (56.41 to 57.81) (`results/phase2/v2/test/final_test.json`); BiomedCLIP probe 50.84%, paired gap -15.2 to -16.1 points, every interval excluding zero (`results/research/b_vlm/results.json`) |
| Evaluation plans: baselines | Zero-shot and linear-probe BiomedCLIP; metadata-only and image-only heads; keyword baseline for clinical text extraction (G5) | Metadata-only 36.4% vs chance 14.3% (`results/research/e_fusion/results.json`) |
| Evaluation plans: splits | Lesion-level splits, frozen across data versions, enforced by a test and a pipeline gate; patient overlap measured where recorded | 47 MSK patients span splits; dropping their test images moves the ensemble from 66.46% to 66.09% (`results/phase2/patient_overlap/patient_overlap.json`) |
| Human review | Grad-CAM review worksheet with a blank reviewer-notes column in Unity Catalog (PR #25); the cohort service runs a definition only after a reviewer approves its exact hash (G2) | 150 images on the worksheet; nothing is called reviewed until a person fills it in (`results/research/d_review/results.json`) |
| Error analysis | Confusion matrices per client and per skin type; worst-miss lists | Commonest error: nevus called melanoma, 199 of 3,262 test images (`results/research/d_review/results.json`) |
| Subgroup and bias assessment | Research Phase C (PR #23): site, Fitzpatrick type, age, sex and body site as OMOP concepts, intervals within groups, two reproduction gates; drift-alert rates by group | Under 30: 27.2% vs 72.3% at 60-74; on BiomedCLIP features a 5% alert threshold fires on 13.3% of MSK images vs 1.4% of HAM10000 Queensland (`results/research/c_subgroups/results_v2.json`) |
| Interpretable approaches | Grad-CAM on the shipped model with the weight-randomization sanity check (Adebayo et al.) and agreement across seeds | Mean rank correlation with a re-initialized network 0.045; 11 maps flagged "not interpreted"; seed agreement 0.81 (`results/research/d_review/results.json`) |
| Reproducible GPU training | Databricks serverless GPU jobs pinned to a commit, with the wheel and every input hash-checked inside the job (`scripts/stage_research_job.py`); 3 seeds; MLflow | BiomedCLIP embeddings on an A10G, job run 1040483728228282, code version and model version recorded (`results/research/b_vlm/embeddings_summary.json`); shipped-model features, job run 498023462798649, checked row by row against the committed predictions (`results/research/e_fusion/shipped/features_summary.json`); retraining loop on an A10: challenger +6.2 points at a new hospital but -1.6 on the original sites, so the pre-registered rule kept the champion (`results/phase3/retrain/decision.json`) |
| Maintainable services with tests | FastAPI on CPU (`app/api`), Docker image built and smoke-tested in CI with an untrained test bundle (`.github/workflows/docker.yml`), ruff, mypy strict and pytest on every push, a CI accuracy gate (`tests/test_eval_gate.py`) | Review workload at the abstain threshold: 5.0% validation, 6.6% test, 17.9% and 30.4% at the new hospitals (`results/phase3/serving/results.json`) |
| Monitoring | Monthly Lakeflow drift job (PR #34): KS and PSI against a validation baseline, per-skin-type alerts, delayed accuracy, an audit table | 12-month replay: no alert in 4 in-distribution months, alerts on every fully shifted month (PSI 0.62 to 4.01) (`results/phase3/monthly/`) |
| Docs | Model card in CHAI Applied Model Card v0.2 structure with an HTI-1 crosswalk; change-control plan mapped to FDA's PCCP guidance; data card; data dictionary; runbook | `docs/model-card.md`, `docs/change-control-plan.md`, `docs/DATA_CARD.md`, `docs/data_dictionary.md` (all "structured after, not compliant with") |
| Governance evidence | Hash-chained audit log for the cohort service; small-count suppression with a brute-force test; Unity Catalog model registry and lineage (PR #35); per-run cost tracking (PRs #32, #38); public-API privacy conditions (no image stored or logged, verified in CI) | Model `workspace.cdm.skin_lesion_classifier` v1 reloads from the registry and matches the bundle on 3 made-up images (`results/phase3/lineage/`); a gold cost table from the system billing tables (34.3 GPU and 26.7 CPU DBUs billed so far, $31.62 at list price, $0 charged on Free Edition); 13 of 31 results traced to their job run's cost so far (`results/phase3/costs/`) |
| Research communication | A short technical report (`docs/report/report.tex`, LaTeX, not yet compiled); README with results first and limitations measured; a public cohort demo with a walkthrough script (`docs/demo_walkthrough.md`) | `docs/report/report.tex`; https://cdm-cohort-demo.streamlit.app/ |

## What is not done, stated plainly

See [docs/BUILT_VS_PLANNED.md](BUILT_VS_PLANNED.md). In short:
- the public API is built but not deployed (on hold pending the owner's decision on public weights);
- clinical text extraction is built but not scored (it waits for hand labels);
- the CMS readmission model (Phase 6) is deferred: there is no monthly history of the CMS data
  to train on or monitor (see the PLAN decisions log, 2026-10-23);
- ONNX serving failed parity and was closed;
- lineage does not yet reach the audit table.
