# Clinical Imaging Drift Monitor Plan

Sep 27, 2026 · Erjon Brucaj

Turn the ECE 570 notebook into a deployed system that flags when a skin lesion classifier sees data it was not trained for. Fix the science first, then add the data, model, serving and LLM layers in phases 0 to 6, about seven weeks of evenings, at zero cost.

## Where it stands

The notebook proves you can fine-tune a model in PyTorch and score drift detectors with the right metrics. Its numbers are not yet defensible, so Phase 0 comes before anything else.

| Piece | Today | What an interviewer would say |
| --- | --- | --- |
| Classifier | EfficientNet-B0 fine-tuned on 10% of HAM10000, 3 epochs, CPU | No held-out accuracy reported, so nobody knows if it learned anything |
| Mahalanobis, HAM vs CIFAR-10 | AUROC 4.1% | Below 50% means the score points the wrong way. In-distribution samples were scored against their true class, CIFAR against the nearest class, and in-distribution scoring used training data |
| Mahalanobis, HAM vs PathMNIST | AUROC 43.4% | PathMNIST ran through the ImageNet model, not the fine-tuned one, and PCA was refit on PathMNIST, so the two sides live in different spaces |
| Federated, FEMNIST | FedAvg 71.9%, Gram 81.3% AUROC | Client models reached 5 to 12% accuracy, and weights were averaged once, not over rounds |
| Federated, HAM | AUROC 62% | Each client held one class, so every client scored 100% and learned nothing useful |
| Code | One notebook, TensorFlow and PyTorch mixed | Not runnable end to end, no tests, no seeds beyond a few |

The strongest honest result today is the FEMNIST comparison: Gram matrix scoring beat Mahalanobis after FedAvg. Everything else gets rerun.

## Target system

The end state is a small production-shaped system: a lakehouse feeds training, the model is served behind an API that returns a drift score with every prediction, and a monthly job writes a grounded incident report when a new site's data drifts.

Flow:

1. Source images (HAM10000 + ISIC archive)
2. Lakehouse (Databricks, bronze to gold, with gates)
3. Training (PyTorch, Flower federated learning, MLflow)
4. Registry + ONNX (versioned model, fast CPU inference)
5. Scoring API (label, OOD score, abstain flag)
6. Monthly drift job (new batches vs baseline)
7. LLM incident report (grounded in drift metrics)
8. Dashboard + runbook (Streamlit, one-command demo)

ISIC does not reliably record the contributing institution (`attribution` is often "Anonymous"), but each image belongs to collections, and several collections come from one hospital. A **site** is therefore a named set of collections in `config/sites.json`. Sites become the federated clients, and held-out sites play the new hospital.

## Data and drift sources

Real drift is the headline. Staged drift exists only to score the detector against a known answer, and the README labels it as staged.

| Source | Real or staged | What it shows |
| --- | --- | --- |
| ISIC Archive, replayed by site month by month | Real data, chosen order | Shifts that already exist: other hospitals, clinical photos vs dermoscopy, more skin types than HAM10000 |
| ISIC Archive, new images checked monthly | Real and live | Whatever contributors add. Months with nothing new are logged as no new data |
| Built-in shifts on known dates (blur, brightness, age mix) | Staged | Time to detect and false alarm rate, because the start date is known |

The ISIC Archive is public in AWS S3 with no account needed, grows when contributors add data, and carries license, attribution, collections, age, sex, body site, image type and, for about 2% of images, Fitzpatrick skin type per image ([AWS registry](https://registry.opendata.aws/isic-archive/), [example image](https://api.isic-archive.com/images/ISIC_6589782/)). How often new images arrive is not published.

## Stack

Databricks Free Edition is the center, lined up with the Data Engineer Associate and ML Associate certifications. Everything stays free.

| Layer | Tool |
| --- | --- |
| Storage and ingest | Unity Catalog volumes and Delta tables, Auto Loader for new images |
| Bronze to gold | Lakeflow Declarative Pipeline in PySpark, with expectations as the data quality gates |
| Scheduling | Lakeflow Jobs (monthly drift run) |
| Training | PyTorch on Databricks serverless GPU (1x A10G, confirmed 2026-09-30), Colab Pro as backup |
| Tracking and registry | MLflow, model registered in Unity Catalog |
| Federated learning | Flower |
| Serving | ONNX Runtime behind FastAPI in Docker, public on Render |
| LLM reports | Gemini free tier, groundedness check from the governed agent |
| UI | Streamlit Community Cloud, calling the API |
| Tests and CI | pytest and GitHub Actions |

Source: [Databricks Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).

## Phase 0: Fix the science (about 1 week)

Every later phase builds on these numbers, so they have to survive a skeptical reviewer first.

- [x] Split train, validation and test by lesion, not by image. HAM10000 has several images of the same lesion, so a random split leaks.
- [x] Train on the full dataset locally on CPU (an overnight run), 10 to 15 epochs, class-weighted loss. Report balanced accuracy and per-class recall on the test split.
- [x] Fit PCA and Mahalanobis statistics on training features only. Score held-out in-distribution images and every OOD set the same way, against the nearest class mean.
- [x] Run every OOD set through the same fine-tuned model and the same PCA.
- [x] Add two standard baselines: maximum softmax probability and energy score.
- [x] Run 3 seeds per result and report the mean and range.
- [x] Move from one notebook to PyTorch scripts (data, train, ood, eval) with tests that the splits never share a lesion and that scores point the right way.

**Done when** one command reproduces every number within the reported run-to-run variation (run locally on CPU), and a short note explains why the old 4.1% AUROC was wrong.

## Phase 1: Data engineering (about 1 week)

Images get the same medallion treatment as reckoner, so the pipeline is the data engineering proof, not a side step.

- [x] **Bronze:** raw HAM10000 and ISIC images and metadata as delivered, in a Unity Catalog volume, with a manifest and checksums (the reckoner pattern). Auto Loader picks up new images. (Built: ingest is a job task that lands files in the volume, and Auto Loader reads them into the three bronze tables; in the one successful update it read all 22,657 image files. Picking up only new files on a later update has not been exercised yet; Phase 3's monthly drift job is its first real test.)
- [x] **Silver:** decoded, resized to 224, duplicates removed, lesion-level split wherever the data is used for training, site attached, and skin type attached where ISIC records it, as Delta tables built in PySpark. Sites used only for scoring need de-duplication, not a split. (Built: silver keeps the latest row per image ID; a duplicate file under two IDs fails the `no_duplicate_files` gate rather than being silently dropped.)
- [x] **Gold:** an embeddings table (image, model version, feature vector), per-site and per-skin-type feature statistics, and the baseline score distributions the drift job compares against. Phase 1 computes embeddings with the pretrained ImageNet EfficientNet-B0 (model version `imagenet-effb0`), because no registered model exists yet; Phase 2 recomputes them with the registered model.
- [x] Build it as one Lakeflow Declarative Pipeline, with expectations as the gates: no lesion in two splits, class counts within expected ranges, no missing labels, failed decodes quarantined rather than dropped. (Built: the pipeline holds bronze, silver and the 11 gates; gold is a job task after it, because it needs torch.)
- [x] Record each image's license and attribution, and keep only images whose license allows this use.

Sources and route, decided after the week 1 Databricks checks:

- **Sites:** Barcelona (BCN20000, collection 249, capped at 5,000 images by whole lesions, chosen with a fixed seed recorded in `config/sites.json` with a fingerprint of the selection), Buenos Aires (HIBA, 251), MSK (287 and 289) and PAD-UFES-20 (406, smartphone photos).
- **HAM10000** comes from ISIC collection 212 by image ID, because Databricks Free Edition cannot reach Harvard Dataverse. ISIC serves these images re-encoded at stronger JPEG compression, so they are not byte-identical to Phase 0's files; the manifest records our own SHA-256 per file. `HAM10000_metadata.csv` is uploaded to the volume once and checked against Dataverse's published MD5, so the Phase 0 lesion split is reproduced exactly (fingerprint `cc2b196cd5bf58ad`). No HAM10000 image may appear in a site table.
- **Ingestion:** a job task downloads each file to `/tmp`, checks it, and copies it into the volume with `shutil.copyfile` (direct writes into a volume failed on large files). Auto Loader then picks up what lands in the volume.
- **Deployment:** a committed job and pipeline definition created with the Databricks CLI, because bundles have not been checked on Free Edition yet.

**Done when** one command rebuilds bronze to gold and a broken input fails a gate loudly.

## Phase 2: ML (1 to 1.5 weeks)

This is where the project becomes real ML work: proper federated training and OOD detection tested on a realistic shift.

- [ ] Train on Databricks serverless GPU (1x A10G: 981 images/s synthetic and 397 images/s on real silver images in the check run, `results/phase2/gpu_check.json`), with Colab Pro as the backup. Log every run to Databricks MLflow and register the chosen model in Unity Catalog.
- [ ] Recompute the gold embeddings and baseline score distributions with the registered model.
- [ ] Tune the centralized model on validation only (seed 0, one change per run, kept only if pooled validation balanced accuracy rises by at least 1 point without hurting macro AUROC or melanoma sensitivity): larger stored images with random resized crops, rotations and colour jitter; about 25 epochs with warmup, cosine learning rate and label smoothing; EfficientNet-B3 at 300 px; test-time flip averaging. Test is scored once, at the end. Federated runs use the winning configuration.
- [ ] Barcelona and MSK get lesion-level train, validation and test splits in silver, with gates; HAM10000's split must stay at fingerprint `cc2b196cd5bf58ad` (gate shown passing). At every site except HAM10000, "Squamous cell carcinoma, NOS" becomes unlabelled and scoring-only, because it can be invasive SCC, which HAM10000's akiec class does not include; HAM10000 keeps its own labels.
- [ ] Federated training with Flower: one client per site (HAM10000 Vienna, HAM10000 Queensland, Barcelona, MSK), 20 or more rounds, FedAvg against FedProx. Centralized and federated models are compared on the same pooled test set: the test splits of all four clients.
  - How it runs (decided 2026-10-02): aggregation uses Flower's own FedAvg and FedProx strategies (`flwr==1.39.0`), but the four clients train one after another in one process on one GPU, not through Flower's Ray simulation engine. Each round's model is scored on pooled validation and the best round is kept, the same rule as centralized training's best epoch; a real federated deployment could not see pooled validation data, so this selection is a simulation convenience, and it is stated wherever federated results appear.
- [ ] v2 data (decided 2026-10-04): Barcelona's cap rises from 5,000 to 10,000 images by the same seeded whole-lesion selection (fingerprint `4129a90a0e5f3aed`, 9,999 images), so v1's 5,000 are a subset. Every v1 lesion keeps its split (`config/splits_v1.csv`, gate `frozen_splits_kept`); only new lesions are assigned; HAM10000's fingerprint is unchanged. The pipeline update that lands the new images is also Phase 1's deferred check that Auto Loader picks up only new files.
- [ ] v2 models: 3 centralized and 6 federated runs (FedAvg and FedProx, 3 seeds each) with v1's configuration and FedAvg's standard image-count weighting; each client's share of training images and aggregation weight, and the bordered share, are reported. The keep rule is applied once, on the centralized models: v2 replaces v1 only if, on v1's exact validation images (mean of 3 seeds), pooled balanced accuracy rises by at least 1 point and no client's falls by more than 3 points, decided and committed before any v2 test look. Whichever data version wins is used by all three setups in the drift table, so the centralized vs federated comparison is not mixed with a data change; federated v1 vs v2 validation results are reported for information. Test: one look per v2 model, compared with v1 on v1's exact test images, with the full v2 test set as a second, separately labelled row. The drift table runs once, on the final models.
- [ ] OOD on held-out sites (real): Buenos Aires (new hospital) and PAD-UFES-20 (new hospital, smartphone photos); PathMNIST and CIFAR-10 as benchmark sets. Held-out accuracy is reported only on classes that map cleanly to HAM10000's (not "SCC, NOS"); PAD-UFES-20 has no df or vasc images.
- [ ] One results table: 4 detectors (Mahalanobis, Gram, max softmax, energy) by shift by centralized vs federated, AUROC and FPR@95TPR, mean and range over 3 seeds, with the run-to-run spread. Mahalanobis's PCA size is chosen on validation data only.
- [ ] Accuracy and drift reported per Fitzpatrick skin type where a site has at least 100 images of that type; smaller groups are listed with counts only. Skin type is recorded only at Buenos Aires and PAD-UFES-20, with 98 images of types IV to VI in total, too few to judge darker skin.
- [ ] Label delay: the monitor watches drift without labels first, then checks accuracy once diagnoses for that batch are released.
- [ ] Error analysis: confusion matrix, which classes, sites and skin types fail, and a look at the 20 worst misses.
- [ ] Risk-coverage and abstain workload: for the shipped model and detector, accuracy and balanced accuracy against coverage when the least confident or most out-of-distribution images are abstained on, per site, and the share of images sent to human review at the abstain threshold Phase 3 will use. Computed on validation to set the threshold, then reported once on test and the held-out sites.

**Done when** the results table is fully logged and you can say which detector to ship and why. The rule, fixed before any Phase 2 result: ship the detector with the highest mean AUROC on the two real held-out sites, with FPR@95TPR breaking ties; the benchmark sets are secondary.

## Phase 3: Serving and MLOps (about 1 week)

The model becomes a service with a threshold, a schedule and a CI gate, which is what production ML and forward deployed roles look for.

- [ ] Export to ONNX, confirm outputs match PyTorch within tolerance, and benchmark ONNX Runtime against PyTorch on CPU (p50 and p95 latency).
- [ ] FastAPI `/predict` returns the label, class probabilities, the OOD score, and an abstain flag when the score passes a threshold chosen on validation data.
- [ ] Docker image, deployed free on Render.
- [ ] Monthly Lakeflow Job: pulls the month's replayed and new ISIC images, scores them, compares the score distribution to the gold baseline (population stability index and a KS test), writes a gold drift table, and flags past a set threshold.
- [ ] Per-skin-type drift alerts: the monthly job also computes PSI and KS per Fitzpatrick type wherever that month has at least 100 images of the type (the Phase 2 rule); smaller groups are logged with counts only and never alert.
- [ ] Prediction audit table: every prediction from the monthly job and the API is written to a gold audit table (request ID, model version and registry URI, input SHA-256, label, class probabilities, OOD score, abstain flag, timestamp). The API appends to a volume file that Auto Loader ingests, so the free Render tier needs no database connection.
- [ ] Unity Catalog lineage shown end to end: bronze → silver → gold → registered model → audit table, with a screenshot and the lineage query in the runbook.
- [ ] Per-run cost tracking: GPU DBU and run time for every job run, read from the system billing and jobs tables and written to a gold cost table, so each result can be traced to what it cost.
- [ ] Staged shifts scored for time to detect and false alarm rate.
- [ ] Retraining loop: a confirmed drift trains a challenger, which replaces the current model in the Unity Catalog registry only if it wins on held-out data.
- [ ] CI: pytest on every push plus a small eval gate that fails the build if AUROC drops, copied from the governed agent's setup.

**Done when** the API has a live URL, the drift job has one green scheduled run, and CI blocks a regression.

## Phase 4: LLM layer (about 1 week)

An LLM turns a drift alarm into a report a clinical ops lead can act on, and every number in it is checked against the data.

- [ ] When the drift job flags, Gemini on the free tier writes an incident report in 8D structure: the problem (what shifted and which classes), containment (hold or abstain), root cause, the permanent fix (relabel a sample or retrain), how the fix was checked, and how to prevent a repeat.
- [ ] Groundedness check: every number in the report must match the drift table, reusing the governed agent's groundedness check.
- [ ] Eval: 20 to 30 synthetic drift scenarios with known findings, scored for correctness and groundedness, with cost and latency tracked per call.
- [ ] Reports land in a human review queue before they count as sent.

**Done when** the eval passes a threshold you set in advance and every report traces back to rows in the drift table.

## Phase 5: Forward deployed packaging (a few evenings)

This phase makes the project easy for someone else to pick up and run, which is the core forward deployed skill.

- [ ] One-command demo (`docker compose up`) with a sample batch that trips the drift alarm.
- [ ] Runbook in 8D steps: containment when the alarm fires, who reviews the report, root cause, retraining as the permanent fix, and the check that prevents a repeat.
- [ ] Model card in the CHAI Applied Model Card format, plus the data card: intended use, data provenance, per-site and per-skin-type results, known failure modes, and the drift monitoring that applies.
- [ ] One-page change-control plan mapped to FDA's final guidance "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions" (December 2024; Federal Register notice of availability, December 4, 2024): the description of modifications (which retraining triggers are allowed), the modification protocol (data, retraining, the champion/challenger test from Phase 3, update and rollback), and the impact assessment. Labelled as a portfolio exercise, not a regulatory submission.
- [ ] README in the reckoner style: results table first, real vs staged drift labeled, a silent-failures list, and a built vs planned page.
- [ ] Optional: a 3-minute demo recording.

## Phase 6: CMS readmission drift (committed, about 1 week, after Phase 5)

This phase is part of the project, not optional: the project is done only when Phase 6's "Done when" is met.

The drift monitor is built to work with any model, and a second model on real monthly CMS data ties the project to your healthcare data background.

- [ ] Train a small readmission model on the CMS hospital quality data your governed agent already refreshes monthly, using its Type 2 history table.
- [ ] Point the same drift job at it: feature drift each month, and performance once the next refresh arrives.
- [ ] Show both models side by side in the Streamlit app.

**Done when** one monthly run scores drift for both models and the README explains what each one shows.

**Left out on purpose:** Kubernetes (Databricks and Render already cover deployment, and a free managed cluster is hard to find), and NER or de-identification (a separate project).

## What each lane gets

Each phase unlocks a claim you can put on a resume and defend in an interview. The numbers stay blank until the runs produce them.

| Lane | Unlocked by | Claim once done |
| --- | --- | --- |
| Data engineering | Phase 1 | Built a medallion image lakehouse with lesion-level, leak-free splits and write-audit-publish gates, published to Databricks Delta |
| ML | Phases 0 and 2 | Trained EfficientNet (B0 or B3, chosen on validation) on HAM10000 and ISIC and federated it across sites with Flower, comparing 4 OOD detectors on a real site shift, with results by skin type (AUROC to fill in) |
| MLOps | Phase 3 | Served the model through ONNX Runtime and FastAPI with a validated abstain threshold, a monthly drift job, and a CI accuracy gate |
| LLMOps and applied AI | Phase 4 | Generated grounded drift incident reports from an LLM, scored on a scenario eval with cost and latency tracked |
| Forward deployed | Phases 3 and 5 | Shipped a one-command demo and runbook a clinical ops team could operate |
| Healthcare data | Phase 6 | Pointed the same drift monitor at a CMS readmission model on monthly public hospital data, with feature drift each month and performance checked once the next refresh arrives |

## Open questions

- [ ] Test in week 1 whether Databricks Free Edition can read the public ISIC S3 bucket, since custom storage locations are not supported.
- [ ] Check whether Free Edition includes Asset Bundles and data quality monitoring before planning around them.
- [ ] Find out whether ISIC images carry an upload date to sort by, and how often new data actually arrives.
- [ ] When does this start? The CM study targets Oct 2 and reckoner's first scheduled run is Oct 1.
- [x] Purdue's policy allows this coursework in a public repo (confirmed Sep 27).
- [ ] Free Edition and CC-BY-NC images are both non-commercial only, which fits a portfolio. Attribution is required for CC-BY images.
