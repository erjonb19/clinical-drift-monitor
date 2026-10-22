# Research extension plan (Phases A to F)

Drafted 2026-10-20 · status: **proposal, awaiting approval. No code is written yet.**

This plan adds six phases aimed at research-informatics work: OMOP/SNOMED gold tables, a
vision-language baseline, subgroup evaluation, interpretability with a human-review worksheet,
multimodal fusion, and a cohort-discovery contract. It follows CLAUDE.md. Every number comes from
a committed run that is also logged in MLflow, staged drift is labelled as staged, and every
limitation is written down.

## Before anything: how this fits PLAN.md

CLAUDE.md makes PLAN.md the source of truth and forbids features that are not in it. Phases A to
F are not in PLAN.md, so approving this plan also means approving a PLAN.md change. Proposed:

1. Add a section "Research extension (Phases A to F)" to PLAN.md, linking here, with each phase's
   checkboxes and "Done when". Make it its own commit before any code.
2. **Order against Phase 3 (your call).** Phase 2 has three open items: per-skin-type accuracy
   (PLAN.md line 117), error analysis with the 20 worst misses (line 119), and the Buenos Aires
   recalibration (line 122, which has run but is not ticked). Phases 3 to 6 have not started. My
   recommendation is to close Phase 2 first, then run A to F before Phase 3, because:
   - Phase C absorbs line 117 and Phase D absorbs line 119, so no work is duplicated.
   - Nothing in A to F depends on Phase 3.
   - Phase 6 is marked "committed", and it keeps its place after Phase 5.
3. **Per-phase workflow change.** Until now commits went straight to `main`. From here on, each
   phase gets a branch `research/<letter>-<slug>` and a PR, merged only with CI green.
4. **Stale front-page docs.** README.md ("Phases 1 to 6 have not started"),
   `docs/BUILT_VS_PLANNED.md` (last updated 2026-09-29) and `STATUS.md` are all behind Phase 2.
   The rule "short README section per phase" would build on a wrong page. I propose one small
   PR before Phase A that only brings these three files up to date from committed evidence, with
   no new numbers.

### Prerequisite shared by B, C, D and E: analysis look, part B

The committed analysis spec (`results/phase2/analysis_look/spec.json`, part B) saves per-image
class probabilities for the shipped v2 ensemble and the three v2 centralized models. It covers
the clients' test split (3,262 images) and every Buenos Aires and PAD-UFES-20 image. Its job
definition is committed (`databricks/analysis_look_job.json`). Only part A has results so far.
Phases B to E need those per-image outputs to compute bootstrap intervals, subgroup metrics and
the worst-miss lists without another look at the test set. **Part B runs first, under its
existing spec and lock.**

The test-look rule stays in force. Any new model that is scored on test gets a spec committed
before it runs and its own lock. These models are BiomedCLIP zero-shot, the BiomedCLIP linear
probe, and the fusion heads. Prompts, probe settings and fusion settings are chosen on
validation only.

## Step 0 checks

### (a) Are splits grouped by lesion? Yes. A second, patient-level issue is open.

**By lesion: yes, enforced in three places.**
- `cdm.splits.lesion_split` assigns whole lesions, stratified by diagnosis.
- `assign_splits` applies that per client source, and frozen v1 lesions keep their split.
- The pipeline gate `no_lesion_in_two_splits` (`cdm.gates.lesions_in_one_split`) fails the
  update on any leak.
- `tests/test_splits.py` and `tests/test_frozen_splits.py` test this in CI. HAM10000's split is
  also pinned to fingerprint `cc2b196cd5bf58ad` by a gate.

No fix is needed for lesion-level grouping.

**By patient: not grouped, and leakage is measurable at MSK.** Splits are by lesion, and one
patient can have several lesions. I counted `patient_id` per ISIC collection from the cached
label-provenance pages (`results/phase2/label_provenance/raw/`, fetched for Phase 2). This was a
one-off count, not yet a committed script:

| Collection | Role | Images | patient_id recorded | Patients | Lesions |
| --- | --- | ---: | ---: | ---: | ---: |
| 212 HAM10000 | client | 11,720 in collection | 0 | none | 8,838 |
| 249 Barcelona | client | 18,946 in collection | 0 | none | 5,440 |
| 287 MSK | client | 2,050 | 0 | none | 1,877 |
| 289 MSK | client | 1,678 | 1,678 | 780 | 923 |
| 251 Buenos Aires | scoring only | 1,616 | 1,616 | 623 | 1,246 |
| 406 PAD-UFES-20 | scoring only | 2,298 | 2,298 | 1,373 | 1,891 |

For MSK collection 289, matched to the committed v1 lesion splits (`config/splits_v1.csv`; MSK
lesions are identical in v2):
- 47 of 730 patients have lesions in more than one split, covering 139 lesions.
- 30 of the 130 test lesions from collection 289 belong to a patient who also has a train lesion.

This is not the same-lesion leak the gate catches. It is a weaker patient-level correlation:
same skin, same camera session, and for Phase E the same age and sex. HAM10000, Barcelona and
MSK collection 287 record no patient ID, so patient-level leakage there **cannot be checked at
all**.

Patient-level grouping also touches Phase 2 results that are already committed:
- The Buenos Aires recalibration split lesions 50/50, and 199 of its 623 patients have more than
  one lesion. Patients can therefore sit in both halves, so its threshold-transfer result is
  somewhat optimistic.
- Bootstrap intervals resampled by lesion (`cdm.final.bootstrap_by_lesion`, `cdm.triage`) are
  too narrow at sites where lesions cluster within patients.

**Options. You decide before Phase A.**
1. *(Recommended)* Keep the frozen splits, so every Phase 2 number stays valid. Record the issue
   in `docs/silent-failures.md` and in the limitations of the Phase 2 note. From Phase B on:
   - resample bootstrap intervals by patient where `patient_id` exists, and by lesion elsewhere;
   - add one sensitivity row per test result that drops the 30 affected MSK test lesions;
   - add a gate counting patients in more than one split, which reports 47 today and is allowed
     to be nonzero only with this documented reason.
2. Re-split MSK collection 289 by patient as data v3. That changes the frozen MSK split, so all
   Phase 2 training, drift and triage results would need rerunning on GPU. That is several
   evenings of compute, and the v1-vs-v2 comparison story is lost.

### (b) Licenses for new models, libraries and data

| Item | Used in | License | Verified how | Note |
| --- | --- | --- | --- | --- |
| BiomedCLIP `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | B, C | **MIT** | `LICENSE.md` read at revision `9f341de24bfb00180f1b847274256e9b65a3a32e`; HF metadata says `mit` | Model card: "intended to be used solely for (I) future research on visual-language processing … Any deployed use case of the model — commercial or otherwise — is currently out of scope." It fits a research benchmark. **It must never go behind the Phase 3 API.** Weights are a pickled `.bin`, so pin the revision. |
| `open_clip_torch` | B | MIT | `LICENSE` file in mlfoundations/open_clip (the GitHub API reports NOASSERTION, but the file is MIT) | The card pins `open_clip_torch==2.23.0` and `transformers==4.35.2` on Python 3.10. Our Python is 3.11, so the version pair is a risk to check in B's first run. |
| `transformers` (tokenizer) | B | Apache-2.0 | GitHub API | Pulled in by open_clip for the PubMedBERT tokenizer. |
| `pytorch-grad-cam` | D, not proposed | MIT | GitHub API | Recommendation: write Grad-CAM directly with torch hooks (about 50 lines, tested), so no dependency is needed. |
| OMOP CDM v5.4 table definitions | A | Apache-2.0 (OHDSI) | OHDSI GitHub | Tables are re-implemented from the spec; no DDL is copied. |
| OMOP medical imaging extension (`image_occurrence`, `image_feature`) | A | Apache-2.0 | OHDSI/ImageWG repo license; column spec in `rmd/Approach.Rmd` | This is a **workgroup proposal**, not part of a released CDM version. Park et al., JAMIA 32(10):1533 (2025), [doi:10.1093/jamia/ocaf091](https://doi.org/10.1093/jamia/ocaf091). |
| OMOP standard vocabularies (Athena) | A | Per vocabulary | Not yet | Free account at athena.ohdsi.org. **Athena's API returned 403 to a scripted request**, so lookups go through the Athena website or a vocabulary download. |
| SNOMED CT (codes inside OMOP) | A | SNOMED International Affiliate License; free in the US through NLM, a member | NLM pages | **Not verified:** whether a public repo may hold SNOMED codes and preferred terms for about 30 concepts. Many OHDSI repos do. Phase A reads the license clause on redistribution before the mapping CSV is committed. If it is unclear, the repo commits only OMOP `concept_id`s (OHDSI identifiers), and the SNOMED codes and terms are fetched from Athena at build time. |
| Fitzpatrick17k | follow-up only, not added | CC BY-NC-SA 3.0 | README in mattgroh/fitzpatrick17k | Images come from two web atlases, so per-image rights are less clear than ISIC's. Listed in Phase C as a follow-up only. |

All of these are free, so zero cost holds. PyTorch stays the only ML framework: open_clip is
PyTorch.

---

## Phase A: OMOP and SNOMED CT gold tables (highest priority)

**What changes**
- **Concept mapping.** A committed mapping CSV with one row per source value:
  - the 7 HAM10000 `dx` codes, plus each ISIC `diagnosis_3` value present at the sites;
  - sex values; body sites (HAM `localization` and ISIC `anatom_site_1`);
  - `dx_type` and ISIC `diagnosis_confirm_type`;
  - modality (dermoscopic vs clinical photo).

  Each row holds the OMOP `concept_id`, vocabulary, concept code, concept name, the Athena URL
  it was looked up at, the lookup date, and a `verified` flag. Unverified rows map to
  `concept_id = 0`, OMOP's "no matching concept", and are listed in the data dictionary. No
  concept ID is entered from memory.
- **Known mapping problems, flagged now.** Three HAM classes are composites with no single
  SNOMED concept:

  | HAM class | What it combines |
  | --- | --- |
  | `akiec` | actinic keratosis + intraepithelial carcinoma (Bowen's disease) |
  | `bkl` | seborrheic keratosis + solar lentigo + lichen-planus-like keratosis |
  | `vasc` | angiomas, angiokeratomas, pyogenic granulomas, haemorrhage |

  The plan maps each image from ISIC's finer `diagnosis_3` where ISIC records it. HAM10000 images
  have it through collection 212. The class-level mapping is the fallback: the lowest common
  parent concept, marked as such. `condition_source_value` always keeps the original `dx`.
- **Tables.** OMOP-style tables built from `silver_metadata`. Default scope: all labelled
  sources, with HAM10000-only views on top. The alternative is HAM10000 only, as written in your
  request. All sources give Phase F a richer cohort.

  | Table | One row per | Contents |
  | --- | --- | --- |
  | `person` | person | `gender_concept_id`; `year_of_birth`, derived (see below) |
  | `condition_occurrence` | lesion | dx mapped to a standard concept; `dx_type` as provenance |
  | `procedure_occurrence` | imaging event | the dermoscopy procedure; required as the parent of `image_occurrence` |
  | `image_occurrence` | image | MI-CDM fields: modality, anatomic site, `local_path` = the volume path |
  | `observation` | lesion | only where a value has no column elsewhere, such as `dx_type` |
  | `concept` | concept used | the subset of concepts used, so joins and the dictionary resolve |
  | `cdm_source` | build | build metadata |

- **How `person` is derived (an assumption, documented).**

  | Source | person derived from | Effect |
  | --- | --- | --- |
  | HAM10000, Barcelona, MSK 287 (no patient ID) | one person per `lesion_id` | over-counts people; a person-level count is a lesion count |
  | MSK 289, Buenos Aires, PAD-UFES-20 | `patient_id` | true patients |

  `person_source_value` records which rule applied to each row.
- **Dates (a limitation, documented).** OMOP requires `year_of_birth`,
  `condition_start_date` and `image_occurrence_date`. HAM10000 and ISIC give no acquisition
  dates, and ages are in 5-year bins. The tables use a single fixed **placeholder** date. Every
  date column is documented as "not a real date". `year_of_birth` = placeholder year − age, and
  is equally artificial. These columns are kept out of all analyses.
- **Missing DICOM identifiers.** `image_study_UID` and `image_series_UID` are required, but
  there is no DICOM. They are filled with deterministic surrogates from `isic_id`, also
  documented.
- **Diagnosis provenance.** `dx_type` and `diagnosis_confirm_type` (histo, follow_up, consensus,
  confocal) are kept as source values. They are mapped to a standard concept where one exists
  (for example, a histopathology procedure), and otherwise sit in `observation`. The exact OMOP
  placement is a judgement call, written into the data dictionary. `condition_type_concept_id`
  records the data's origin (a registry/research type concept, looked up, not guessed).
- **Data quality checks.** Pure-pandas functions that run as tests in CI on synthetic frames,
  and as pipeline gates on real data:
  - expected row counts per table;
  - null rates within stated limits;
  - every `dx` mapped (`concept_id != 0`, or a flagged composite);
  - primary keys unique;
  - foreign keys resolve;
  - one condition per lesion;
  - the mapping reproduces the 7 HAM labels exactly, the same idea as the existing
    `mapping_reproduces_ham_labels` gate.
- **Lineage.** The tables are built inside the existing Lakeflow pipeline, so Unity Catalog
  lineage shows bronze → silver → OMOP. Proof is a lineage screenshot and the
  `system.access.table_lineage` query, both in the runbook.
- **MLflow.** One run per build logs the table row counts, null rates and gate results. There is
  no model.
- `docs/data_dictionary.md`: every table and column, giving type, source column, transform,
  null rate (filled from the run), and caveats. It also states FAIR alignment honestly:
  - **Findable** through Unity Catalog comments and tags;
  - **Accessible** in-workspace only, with no public endpoint;
  - **Interoperable** through OMOP and SNOMED;
  - **Reusable** with license and attribution per image in `image_occurrence`.

**Files**

| Action | File |
| --- | --- |
| new | `src/cdm/omop.py` (transforms and checks) |
| new | `config/omop_concepts.csv` |
| new | `docs/data_dictionary.md` |
| new | `tests/test_omop.py` |
| extend | `pipelines/lakehouse.py` (OMOP materialized views and an OMOP gate table) |
| extend | `docs/phase1-runbook.md` (lineage query) |
| extend | `docs/DATA_CARD.md` |
| update | README section, `docs/BUILT_VS_PLANNED.md` |

**Databricks tables:** `workspace.cdm_omop.{person, condition_occurrence, procedure_occurrence,
image_occurrence, observation, concept, cdm_source, omop_gate_results}`. Fallback, if the
pipeline cannot publish to a second schema on Free Edition: `workspace.cdm.omop_*`.

**Done when**
- One pipeline update builds every OMOP table with all OMOP gates passing.
- Every `dx` is mapped to a cited, verified concept, or is listed as unverified with the reason.
- The data dictionary covers every column.
- The lineage screenshot shows silver → OMOP.
- The checks run in CI, and the MLflow run ID is recorded.

**Risks**
- Athena lookups are manual (the API returned 403). I can drive the Athena website through the
  Chrome extension, or you download the vocabulary bundle once.
- The SNOMED redistribution question (see licenses above).
- Composite classes may not map cleanly.
- Writing to a second schema from one pipeline on Free Edition is not yet checked.
- ISIC's body-site vocabulary differs from HAM's and needs harmonizing.

**Time:** 5 to 7 evenings.

## Phase B: vision-language baseline (BiomedCLIP)

**What changes**
- **Zero-shot BiomedCLIP** on the same pooled client test split (3,262 images). A HAM10000-only
  row (1,528 images) and the external sites (Buenos Aires, PAD-UFES-20, clean classes only, per
  the analysis spec's mapping) are labelled separately. Prompt templates are chosen on
  validation, then frozen in a committed spec.
- **Linear probe** on BiomedCLIP image embeddings. It trains on the train split, is tuned on
  validation, and runs over 3 seeds. The probe is a torch linear head, not sklearn's
  deterministic solver. That way the seeds actually vary.
- **Metrics:** balanced accuracy and melanoma sensitivity, for BiomedCLIP and for EfficientNet
  (the shipped v2 ensemble and the 3 single seeds). EfficientNet's numbers come from analysis
  look part B outputs, with no new test look. Each comes with a 95% bootstrap interval,
  resampled by patient where known and by lesion elsewhere.
- **Drift on BiomedCLIP features**, against the same OOD sets: Buenos Aires and PAD-UFES-20
  (real), PathMNIST and CIFAR-10 (benchmark sets).
  - Mahalanobis on embeddings, fit on train only, with PCA chosen by Phase 2's method D on
    validation.
  - Max softmax probability and energy from both the zero-shot and the probe logits.
  - **Gram is not applicable as built.** It needs convolutional feature maps, and BiomedCLIP is
    a ViT. That is stated rather than adapted, since adapting it would be a new method.

  AUROC and FPR@95TPR, compared with the Phase 2 EfficientNet table. Everything runs on GPU in
  one notebook, so CPU-vs-GPU score mismatch (silent-failures) cannot mix in.
- **Contamination caveat.** BiomedCLIP was pretrained on PMC-15M figure–caption pairs. Papers
  on HAM10000 and ISIC may include figures of these images, so "zero-shot" cannot be shown to
  be unseen. This goes into every place zero-shot results appear.

**Files**

| Action | File |
| --- | --- |
| new | `src/cdm/vlm.py` (load at pinned revision, embed, zero-shot) |
| new | `src/cdm/probe.py` (heads on frozen embeddings; extended in E) |
| extend | `src/cdm/final.py` (bootstrap clustered by patient where known) |
| new | `databricks/notebooks/vlm_baseline.py`, `databricks/vlm_job.json` |
| new | `tests/test_vlm.py` (tiny fake model, no download in CI; scores point the right way) |
| extend | `pyproject.toml` (optional extra `vlm`) |
| results | `results/research/b_vlm/` (spec first, then results) |

**Databricks tables:** rows appended to `workspace.cdm.gold_embeddings` with model version
`biomedclip-9f341de` (the existing table, not a new one). Results go to MLflow experiment
`/cdm/research`.

**Done when**
- The spec is committed before the test look.
- One MLflow run per model holds the balanced accuracy and melanoma sensitivity intervals and
  the drift AUROC/FPR table, for both models.
- Results are committed. CI is green, including scores-point-the-right-way tests on BiomedCLIP
  features.

**Risks**
- open_clip and transformers version pins on Python 3.11 and the serverless image.
- Hugging Face reachability from serverless. Fallback: download once into the volume and check
  the SHA-256.
- Zero-shot may be near chance on 7 dermoscopy classes. That is a valid result and is reported
  as such.
- The Free Edition GPU quota.

**Time:** 4 to 6 evenings.

## Phase C: subgroup and bias evaluation

**What changes**
- **Groups:** age band (fixed bands from 5-year `age_approx`, set in the spec), sex, and body
  site (harmonized through Phase A's anatomic concepts).
- **Performance per group:** balanced accuracy and melanoma sensitivity for EfficientNet and
  BiomedCLIP (zero-shot and probe), with bootstrap intervals.
- **Drift per group:**
  - Mahalanobis score distributions by group;
  - the in-distribution flag rate per group at the 95%-TPR threshold, meaning how often normal
    patients from a group would be flagged or abstained on;
  - external-site AUROC per group.
- **Small groups:** n < 100 is the repo's existing minimum, and also a per-class minimum for
  melanoma sensitivity. Such groups get counts only and a warning in place of a metric.
- **Reads from OMOP:** metadata comes from Phase A's tables, and predictions from committed
  per-image outputs (CPU only, no new test look).
- **Skin type, stated correctly.** HAM10000 has no Fitzpatrick type. **The repo already
  reports drift by Fitzpatrick type** at Buenos Aires and PAD-UFES-20 (analysis part A,
  `results/phase2/analysis_look/skin_type_drift.json`). Types IV to VI have only 98 images in
  total, too few to judge darker skin. Phase C restates that finding and adds accuracy by type
  where n ≥ 100, which closes PLAN.md line 117. Fitzpatrick17k (CC BY-NC-SA 3.0) is listed as a
  follow-up only and is not added.

**Files**

| Action | File |
| --- | --- |
| extend | `src/cdm/eval.py` (subgroup metrics with the minimum-n rule) |
| new | `scripts/subgroups.py` (CPU, reads committed outputs and OMOP tables) |
| extend | `tests/` (min-n warning, groups partition the data) |
| results | `results/research/c_subgroups/` |

**Databricks tables:** none new. It reads `workspace.cdm_omop.*` and logs to MLflow.

**Done when**
- The spec is committed first.
- One MLflow run holds the full subgroup table with intervals and warnings.
- PLAN.md line 117 is ticked.

**Risks**
- Many site × group cells will fall below n, so the honest output is mostly "too few".
- Multiple comparisons: no significance claims are made, intervals only.
- Body site coding differs across sources.

**Time:** 2 to 3 evenings.

## Phase D: interpretability and a human-review worksheet

**What changes**
- **Grad-CAM** on EfficientNet-B0's last convolutional block, per seed, averaged over the 3
  ensemble seeds.
- **Sanity check** (Adebayo et al. 2018): heatmaps must change when the model's weights are
  randomized. This is a test, so the maps are known to depend on the model and are not just
  edge detectors.
- **Worksheet:** the top misclassifications by confidence (per class, including the 20 worst
  misses, which closes PLAN.md line 119) and the top OOD-flagged images by Mahalanobis score.
  Each row holds:
  - the image and its heatmap;
  - true label, predicted label, probability and OOD score;
  - site, license and attribution;
  - a blank `reviewer_notes` column.

  Plus a confusion matrix.
- **Storage.** CLAUDE.md forbids committing data files, so images and heatmaps live in the volume
  and as MLflow artifacts (an HTML worksheet). The repo commits only the worksheet's CSV of IDs
  and values, and the code. ISIC licenses (CC-0, CC-BY, CC-BY-NC) allow derived heatmaps with
  attribution for non-commercial use, so attribution is printed on every row.

**Files**

| Action | File |
| --- | --- |
| new | `src/cdm/explain.py` (Grad-CAM with hooks) |
| new | `databricks/notebooks/review_worksheet.py`, `databricks/review_job.json` |
| new | `tests/test_explain.py` (shape, randomization sanity check) |
| results | `results/research/d_review/` (CSV only) |

**Databricks tables:** `workspace.cdm.gold_review_worksheet`, with `reviewer_notes` writable
from the Databricks table editor. Heatmaps go in
`/Volumes/workspace/cdm/raw/review/<run_id>/`.

**Done when**
- The worksheet exists with all fields.
- The sanity test passes in CI.
- The MLflow run holds the HTML artifact.
- PLAN.md line 119 is ticked.

**Risks**
- Grad-CAM at 7×7 resolution on 224 px input is coarse. That is stated, and it makes no claim
  of faithfulness beyond the sanity test.
- Reviewer notes need a real reviewer. Until someone fills them in, the worksheet is
  "ready for review", not "reviewed".

**Time:** 3 to 4 evenings.

## Phase E: multimodal fusion

**What changes**
- **Heads:** on frozen image embeddings (EfficientNet penultimate per seed, and BiomedCLIP),
  combined with age, sex and body site, plus missing-value indicators, since 57 HAM images
  lack age and ISIC coverage varies.
- **Fair comparison:** image-only vs image + metadata, using **the same head type on the same
  embeddings**, 3 seeds, same lesion-grouped splits. The end-to-end EfficientNet is reported as
  a reference row.
- **Selection:** on validation, then one test look under a committed spec. Bootstrap intervals
  as in B. External sites are reported separately.
- **Shortcut checks:**
  - metadata alone (no image) as a baseline;
  - fusion with source held out where possible;
  - the MSK patient-overlap sensitivity row. Shared age and sex across a patient's lesions are
    exactly what fusion could exploit.
- Reported as measured, including "no gain" or "gain only from shortcuts".

**Files:** extend `src/cdm/probe.py`; new `databricks/notebooks/fusion.py` and job JSON; extend
tests; `results/research/e_fusion/`.

**Databricks tables:** reads `gold_embeddings` and the OMOP tables. Nothing new, apart from
MLflow.

**Done when**
- The spec is committed first.
- MLflow runs exist for image-only, metadata-only and fusion, with intervals.
- A README section states the result plainly.

**Risks**
- Metadata may encode source and therefore label prevalence, which can inflate fusion.
- Small gains will sit inside the intervals.

**Time:** 3 to 4 evenings.

## Phase F: cohort discovery (plan only, no code)

**What changes:** one document, `docs/cohort_contract.md`. It covers:
- **Table contract.** A read-only view `workspace.cdm_omop.v_cohort_lesion`, one row per lesion,
  with:
  - person, condition and image concept IDs and names;
  - harmonized body-site concept;
  - age band, sex, site, diagnosis provenance;
  - image count, license and attribution.

  Each column has a stated meaning and caveats. Placeholder dates are excluded, and "person" is
  a lesion where patient ID is absent.
- **Interface for governed-clinical-agent.** I read that repo first, as CLAUDE.md asks.
  - Plain English goes to a **fixed set of parameterized query templates**, never free SQL.
  - Allowed filters: diagnosis concept, including descendants through a committed
    `concept_ancestor` subset; body site; age band; sex; site; confirmation method.
  - Results are counts plus lesion IDs.
  - The agent's groundedness check confirms every number in the answer against the query result.
  - Every request is written to an audit table.
  - Small-cell counts are suppressed, for example under 11, as research practice even though
    the data is public.
- **Worked example.** "Melanoma lesions on the back in patients over 60" resolves to:
  - melanoma descendants;
  - posterior trunk / back anatomic concepts;
  - `age_approx` ≥ 65, or 60 depending on how bins are read. This ambiguity must be shown to the
    user, not resolved silently.

  It also lists what the contract cannot answer: dates, true patient counts at HAM10000.

**Files:** `docs/cohort_contract.md` only. **Tables:** the view definition is written down but
not created.

**Done when** the contract is reviewed by you, and the corresponding item in the agent repo's
plan is noted.

**Time:** 1 evening.

---

## Summary

| Phase | New modules | Tables | Time |
| --- | --- | --- | --- |
| Docs refresh + PLAN.md amendment | none | none | 1 evening |
| Prerequisite: analysis look part B | none (spec and job committed) | none | 1 evening, CPU job |
| A: OMOP and SNOMED | `omop.py` | `cdm_omop.*` (8) | 5–7 evenings |
| B: BiomedCLIP | `vlm.py`, `probe.py` | `gold_embeddings` rows | 4–6 evenings |
| C: subgroups | none (extends `eval.py`) | none | 2–3 evenings |
| D: Grad-CAM and review | `explain.py` | `gold_review_worksheet` | 3–4 evenings |
| E: fusion | none (extends `probe.py`) | none | 3–4 evenings |
| F: cohort contract | none | none (view specified) | 1 evening |

Total about 20 to 27 evenings. The estimates are rough and assume no Free Edition surprises.

## Decisions needed from you

1. Approve the PLAN.md amendment and the order: close Phase 2, then A to F, then Phase 3.
2. Patient-level leakage: option 1 (keep splits, document, sensitivity rows; recommended) or
   option 2 (re-split MSK and rerun Phase 2).
3. Phase A scope: all labelled sources (recommended) or HAM10000 only.
4. The docs-refresh PR before Phase A: yes or no.
5. Athena lookups: I drive the Athena website in Chrome, or you download the vocabulary bundle.
