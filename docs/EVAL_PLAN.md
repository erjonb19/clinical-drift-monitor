# Evaluation plan (study protocol)

Version 1, 2026-10-21. Changes are added to the amendment log at the end; earlier text is not
edited after a look it governs.

This is the protocol for every evaluation in this repository. It is a **portfolio study on
public research data** (HAM10000, ISIC Archive collections, PathMNIST, CIFAR-10, and, for
Phase G5, MTSamples transcriptions). There is no IRB, no patient contact and no clinical use.
It is written as a protocol so each result can be traced to a decision made before the data
were looked at. Results live in `results/`, each logged in MLflow
(`results/mlflow_runs.json`). This file holds no results.

## 1. Questions

| # | Question | Phase | Primary endpoint |
| --- | --- | --- | --- |
| Q1 | Does the shipped classifier keep its accuracy at sites it was not trained on? | 2 (done) | Balanced accuracy; melanoma sensitivity at the triage operating point |
| Q2 | Does the drift score flag those sites before labels arrive? | 2 (done) | OOD AUROC, test split vs each site |
| Q3 | Does a general biomedical vision-language model (BiomedCLIP) match the trained model? | B | Balanced accuracy and melanoma sensitivity, same test split |
| Q4 | Do performance and drift differ by age, sex, body site and skin type? | C | Per-group balanced accuracy and melanoma sensitivity |
| Q5 | What do the worst errors and most out-of-distribution images have in common? | D | Reviewer categories on the review worksheet (qualitative) |
| Q6 | Does adding age, sex and body site to image features help? | E | Balanced accuracy: fusion vs image-only |
| Q7 | Can an LLM pull diagnosis, body site and confirmation method from notes into OMOP fields? | G5 | Per-field exact-match accuracy against hand labels |

## 2. Data and units of analysis

- **Sources:** described in `docs/DATA_CARD.md`, with license and attribution per image.
  The OMOP tables (`docs/data_dictionary.md`) are the research-ready form of the same data.
- **Unit of analysis:** the image is scored, but the **lesion** is the unit of independence.
  Where ISIC records a patient ID (MSK collection 289, Buenos Aires, PAD-UFES-20), the
  **patient** is the cluster for intervals.
- **Labels:** HAM10000's own `dx`. At the sites, ISIC's diagnosis is mapped by the site
  label rule (`cdm.sources.map_diagnosis`). Each label's confirmation method is kept
  (histopathology, follow-up, consensus, confocal), because labels confirmed without biopsy
  are weaker.

## 3. Splits

- Splits are by lesion, never by image, and a pipeline gate fails any lesion that crosses
  splits. The v1 lesion splits are frozen (`config/splits_v1.csv`): new data never moves an
  earlier lesion.
- Splits are not by patient. The 47 MSK patients whose lesions sit in more than one split
  are documented (`config/patient_overlap_v1.csv`); any other such patient fails the
  `patients_in_one_split` gate. From the research extension on, every test result also has a
  row without those patients; the first such rows are in
  `results/phase2/patient_overlap/patient_overlap.json`.
- **Held-out sites** (Buenos Aires, PAD-UFES-20) are never used for training, model
  selection, calibration or threshold choice.
- **Validation** picks models, thresholds and PCA sizes. **Test** is scored once per model,
  after the spec is committed (section 6).
- G5 notes are split into a development set (prompt design) and a held-out scoring set
  before any prompt is run on the scoring set.

## 4. Comparators and baselines

| Question | Compared against |
| --- | --- |
| Q1 | Majority class and class-prior baselines; the same model on its own client test split |
| Q2 | Max softmax probability and energy, against Mahalanobis and Gram (same features, statistics fit on training data only) |
| Q3 | BiomedCLIP zero-shot (no training), and a linear probe on BiomedCLIP features (3 seeds), against the shipped EfficientNet ensemble |
| Q6 | Image-only with the same head on the same embeddings; metadata-only (age, sex, site); fusion |
| Q7 | A keyword and regular-expression baseline on the same notes, against Claude Haiku |

## 5. Endpoints and how they are computed

- **Balanced accuracy:** mean per-class recall over the classes present.
- **Melanoma sensitivity and specificity:** at a threshold fixed on validation before the
  test look (the triage operating point: the highest threshold with at least 95% melanoma
  sensitivity on validation).
- **Macro AUROC:** one-vs-rest over the classes present.
- **OOD AUROC and FPR at 95% TPR:** in-distribution test scores against each OOD set.
  Scores are computed the same way on both sides, and a test checks they point the right
  way.
- **G5 extraction:** per field (diagnosis, body site, confirmation method), the share of
  notes where the extracted OMOP concept ID equals the hand-labelled one. A note with no
  such information must be labelled "not stated", and saying "not stated" counts as correct
  for it. Precision and recall are also given for "stated vs not stated".

## 6. Analysis rules

- **Seeds:** every trained model runs with 3 fixed seeds. Results are the mean and range
  over seeds; the shipped model is the 3-seed ensemble.
- **Intervals:** 95% percentile bootstrap with 1,000 resamples, resampling patients where
  recorded and lesions elsewhere (`cdm.final.bootstrap_by_lesion`). G5 resamples notes.
- **Multiplicity:** results are descriptive. No p-values, no claims of significance. Where
  intervals overlap, the text says the difference is not resolved.
- **Test-look discipline:** before any test or held-out site is scored, a `spec.json` with
  `"committed_before_running": true` is committed, naming the models, data, metrics,
  thresholds and groups. Each model's look writes a lock in the workspace volume
  (`databricks/notebooks/analysis_look.py`, `drift_table.py`), and an existing lock refuses a
  second run. A second look at the same data needs
  a recorded amendment and is reported as a second look.
- **Too good or too bad:** an AUROC below 50%, a near-perfect score, or a large gap between
  seeds stops reporting until it is explained. Each explanation goes in the phase note, and
  any bug that reported success goes in `docs/silent-failures.md`.

## 7. Subgroups and bias

- **Groups:** age band (fixed bands over the 5-year `age_approx`, set in the Phase C
  spec), sex, body site, and the site itself. Skin type is analysed only within a site
  (Buenos Aires, PAD-UFES-20); HAM10000 records none.
- **Minimum size:** a group with fewer than 100 images, or fewer than 100 lesions for the
  class in question, gets counts and a warning, never a metric.
- **No pooling across sites** for skin type: site and skin type are confounded.
- **What is reported:** each group's metric, its interval, and its difference from the whole
  test split. A gap is described as a gap, not as bias of a cause, unless an analysis
  separates the cause.
- **Known gaps:** Fitzpatrick types IV to VI are almost absent. Fitzpatrick17k is a listed
  follow-up only.

## 8. Error analysis and human review

- **Worksheet (Phase D):** the worst misclassifications (most confident wrong answers) and
  the most out-of-distribution images. Each row has the image, a Grad-CAM heatmap, the true
  and predicted labels, the scores, the license and attribution, and blank reviewer columns.
- **Reviewer instructions:** categorise each row as label doubt, image quality, artefact (ruler,
  ink, hair, vignette), rare presentation, or model error. The heatmap is shown after the
  first category is chosen, so it does not anchor the choice.
- **Grad-CAM sanity (Phase D, built 2026-10-22):** a weight-randomization test will run in CI,
  and heatmaps that survive
  randomization are not interpreted.
- **G5 errors:** every disagreement with the hand labels is categorised as missed, wrong
  concept, wrong granularity (for example melanoma vs melanoma in situ), hallucinated, or label
  error.

## 9. Reproducibility

- Code is pinned by commit. Every Databricks deploy is named after its commit, and every
  results file records the code version that produced it.
- Every committed results file is logged to MLflow with its SHA-256. CI checks that the
  index matches the files.
- Data identity is checked by fingerprints: the HAM10000 split, the capped Barcelona
  selection, and the per-file SHA-256 manifest.
- CPU/GPU reproducibility tolerances are set before running. A failed tolerance is
  reported, never loosened.
- LLM runs (G5) record the model ID, prompt hash, and token cost per call. The model takes no
  temperature setting, so each answer is cached and a re-run replays it instead of sampling
  again.
  CI never calls an LLM; it replays recorded outputs.

## 10. Failure criteria (decided before the look)

A result is **not reported as a result** if any of these hold. What happened is reported
instead.

1. A data gate failed in the run that produced it.
2. The spec was committed after the look, or a lock shows an earlier look at the same data.
3. A reproducibility tolerance failed, and the result depends on the failing component.
4. An unexplained "too good or too bad" result (section 6).
5. For G5: fewer than 50 hand-labelled scoring notes, or a prompt changed after the scoring
   set was first run.

A method **fails its question** when its interval on the primary endpoint lies wholly
below the comparator's, or, for G5, wholly below the keyword baseline's. That is reported
as a finding, with the same prominence as a success.

## 11. Amendment log

| Date | Change | Why | Looks affected |
| --- | --- | --- | --- |
| 2026-10-21 | Protocol written down from the rules Phases 0 to 2 already followed, plus Phases B to G5 | Phase G1 | None: Phases 0 to 2 were run under the same rules, recorded in their specs |
| 2026-10-22 | G5: notes from any specialty that name a skin-lesion diagnosis, a skin biopsy or a skin lesion are included (`cdm.extract.SPECIFIC`, `GENERIC`), and the development share is 10% of each selection group instead of 20% overall | Owner's decision: MTSamples has 29 Dermatology notes and 1 skin-pathology note (the first sheet), too few for the 50-note scoring minimum; rules and counts in `results/research/g5_extraction/selection_spec.json`, written by `scripts/extraction.py selection-spec` | None: no note had been labelled and no prompt had run on a scoring note |
| 2026-10-22 | G5: no temperature 0; answers are cached and replayed | The API refuses a temperature setting for Claude Haiku 5.5 ("deprecated for this model"), found on the first connection test | None: no G5 note had been run |
