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

ISIC records the contributing institution for each image, so institutions become the federated clients and held-out institutions play the new hospital.

## Data and drift sources

Real drift is the headline. Staged drift exists only to score the detector against a known answer, and the README labels it as staged.

| Source | Real or staged | What it shows |
| --- | --- | --- |
| ISIC Archive, replayed by institution month by month | Real data, chosen order | Shifts that already exist: other hospitals, clinical photos vs dermoscopy, more skin types than HAM10000 |
| ISIC Archive, new images checked monthly | Real and live | Whatever contributors add. Months with nothing new are logged as no new data |
| Built-in shifts on known dates (blur, brightness, age mix) | Staged | Time to detect and false alarm rate, because the start date is known |

The ISIC Archive is public in AWS S3 with no account needed, grows when contributors add data, and carries institution, license, age, sex, body site, Fitzpatrick skin type and image type per image ([AWS registry](https://registry.opendata.aws/isic-archive/), [example image](https://api.isic-archive.com/images/ISIC_6589782/)). How often new images arrive is not published.

## Stack

Databricks Free Edition is the center, lined up with the Data Engineer Associate and ML Associate certifications. Everything stays free.

| Layer | Tool |
| --- | --- |
| Storage and ingest | Unity Catalog volumes and Delta tables, Auto Loader for new images |
| Bronze to gold | Lakeflow Declarative Pipeline in PySpark, with expectations as the data quality gates |
| Scheduling | Lakeflow Jobs (monthly drift run) |
| Training | PyTorch on Databricks serverless GPU after LinkedIn verification, Colab Pro as backup |
| Tracking and registry | MLflow, model registered in Unity Catalog |
| Federated learning | Flower |
| Serving | ONNX Runtime behind FastAPI in Docker, public on Render |
| LLM reports | Gemini free tier, groundedness check from the governed agent |
| UI | Streamlit Community Cloud, calling the API |
| Tests and CI | pytest and GitHub Actions |

Source: [Databricks Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).

## Phase 0: Fix the science (about 1 week)

Every later phase builds on these numbers, so they have to survive a skeptical reviewer first.

- [ ] Split train, validation and test by lesion, not by image. HAM10000 has several images of the same lesion, so a random split leaks.
- [ ] Train on the full dataset locally on CPU (an overnight run), 10 to 15 epochs, class-weighted loss. Report balanced accuracy and per-class recall on the test split.
- [ ] Fit PCA and Mahalanobis statistics on training features only. Score held-out in-distribution images and every OOD set the same way, against the nearest class mean.
- [ ] Run every OOD set through the same fine-tuned model and the same PCA.
- [ ] Add two standard baselines: maximum softmax probability and energy score.
- [ ] Run 3 seeds per result and report the mean and range.
- [ ] Move from one notebook to PyTorch scripts (data, train, ood, eval) with tests that the splits never share a lesion and that scores point the right way.

**Done when** one command reproduces every number (run locally on CPU), and a short note explains why the old 4.1% AUROC was wrong.

## Phase 1: Data engineering (about 1 week)

Images get the same medallion treatment as reckoner, so the pipeline is the data engineering proof, not a side step.

- [ ] **Bronze:** raw HAM10000 and ISIC images and metadata as delivered, in a Unity Catalog volume, with a manifest and checksums (the reckoner pattern). Auto Loader picks up new images.
- [ ] **Silver:** decoded, resized to 224, duplicates removed, lesion-level split, institution and skin type attached, as Delta tables built in PySpark.
- [ ] **Gold:** an embeddings table (image, model version, feature vector), per-institution and per-skin-type feature statistics, and the baseline score distributions the drift job compares against.
- [ ] Build it as one Lakeflow Declarative Pipeline, with expectations as the gates: no lesion in two splits, class counts within expected ranges, no missing labels, failed decodes quarantined rather than dropped.
- [ ] Record each image's license and attribution, and keep only images whose license allows this use.

**Done when** one command rebuilds bronze to gold and a broken input fails a gate loudly.

## Phase 2: ML (1 to 1.5 weeks)

This is where the project becomes real ML work: proper federated training and OOD detection tested on a realistic shift.

- [ ] Train on Databricks serverless GPU (Colab Pro as backup), logging every run to MLflow and registering the chosen model in Unity Catalog.
- [ ] Federated training with Flower: one client per institution, 20 or more rounds, FedAvg against FedProx, compared to the centralized model's accuracy.
- [ ] OOD on three kinds of shift: held-out institutions (real), PathMNIST (near), CIFAR-10 (far).
- [ ] One results table: 4 detectors (Mahalanobis, Gram, max softmax, energy) by 3 shifts by centralized vs federated, AUROC and FPR@95TPR, mean over 3 seeds.
- [ ] Accuracy and drift reported per Fitzpatrick skin type, including which types have too few images to judge.
- [ ] Label delay: the monitor watches drift without labels first, then checks accuracy once diagnoses for that batch are released.
- [ ] Error analysis: confusion matrix, which classes, institutions and skin types fail, and a look at the 20 worst misses.

**Done when** the results table is fully logged and you can say which detector to ship and why.

## Phase 3: Serving and MLOps (about 1 week)

The model becomes a service with a threshold, a schedule and a CI gate, which is what production ML and forward deployed roles look for.

- [ ] Export to ONNX, confirm outputs match PyTorch within tolerance, and benchmark ONNX Runtime against PyTorch on CPU (p50 and p95 latency).
- [ ] FastAPI `/predict` returns the label, class probabilities, the OOD score, and an abstain flag when the score passes a threshold chosen on validation data.
- [ ] Docker image, deployed free on Render.
- [ ] Monthly Lakeflow Job: pulls the month's replayed and new ISIC images, scores them, compares the score distribution to the gold baseline (population stability index and a KS test), writes a gold drift table, and flags past a set threshold.
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
- [ ] Model card and data card: what the model is for, what data it saw, per-skin-type results, and where it fails.
- [ ] README in the reckoner style: results table first, real vs staged drift labeled, a silent-failures list, and a built vs planned page.
- [ ] Optional: a 3-minute demo recording.

## Phase 6: A second model on tabular healthcare data (about 1 week, after Phase 5)

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
| ML | Phases 0 and 2 | Trained EfficientNet-B0 on HAM10000 and ISIC and federated it across institutions with Flower, comparing 4 OOD detectors on a real institution shift, with results by skin type (AUROC to fill in) |
| MLOps | Phase 3 | Served the model through ONNX Runtime and FastAPI with a validated abstain threshold, a monthly drift job, and a CI accuracy gate |
| LLMOps and applied AI | Phase 4 | Generated grounded drift incident reports from an LLM, scored on a scenario eval with cost and latency tracked |
| Forward deployed | Phases 3 and 5 | Shipped a one-command demo and runbook a clinical ops team could operate |

## Open questions

- [ ] Test in week 1 whether Databricks Free Edition can read the public ISIC S3 bucket, since custom storage locations are not supported.
- [ ] Check whether Free Edition includes Asset Bundles and data quality monitoring before planning around them.
- [ ] Find out whether ISIC images carry an upload date to sort by, and how often new data actually arrives.
- [ ] When does this start? The CM study targets Oct 2 and reckoner's first scheduled run is Oct 1.
- [x] Purdue's policy allows this coursework in a public repo (confirmed Sep 27).
- [ ] Free Edition and CC-BY-NC images are both non-commercial only, which fits a portfolio. Attribution is required for CC-BY images.
