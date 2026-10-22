# Clinical Imaging Drift Monitor

A skin lesion classifier trained across hospitals, with an out-of-distribution (drift) score
tested on real new hospitals. Built on Databricks Free Edition at zero cost. A personal
portfolio project and research prototype: **not for clinical use**.

**Status (2026-10-22):**
- Phases 0 to 2 are done: fix the science, data engineering, and ML. The one Phase 2 item left,
  error analysis, moved to research Phase D.
- The research extension (Phases A to F: OMOP data, a vision-language baseline, subgroups,
  interpretability, fusion) is planned in
  [docs/research_extension_plan.md](docs/research_extension_plan.md). Phase A (OMOP tables),
  Phase B (the BiomedCLIP baseline), C (subgroups and bias), D (Grad-CAM review
  worksheet) and E (image plus metadata) are built; Phase G
  (research data services) is in progress.
- Phases 3 to 6 have not started.

A 4-page technical report (LaTeX source with standard packages, written to build on Overleaf;
not yet compiled, since no TeX is installed on the build machine):
[docs/report/report.tex](docs/report/report.tex).

[PLAN.md](PLAN.md) has scope and order. [docs/BUILT_VS_PLANNED.md](docs/BUILT_VS_PLANNED.md)
has what exists today.

Every number below comes from a committed results file named beside it. That file is also
logged as an MLflow run in Databricks. [results/mlflow_runs.json](results/mlflow_runs.json)
maps each file to its run, and a CI test fails if a file changes after it was logged. No drift
in these tables is staged: every shift is a real, separate dataset.

## Results (Phase 2: four hospitals, two held-out hospitals)

**Data.** 27,656 images from the ISIC Archive, landed through a Databricks medallion pipeline
with data-quality gates that fail the update (15 deployed; the patient-level gate added on 2026-10-20 deploys with Phase A):
- training clients: HAM10000 (Vienna and Queensland), Barcelona and MSK;
- held out as new hospitals: Buenos Aires and PAD-UFES-20 (smartphone photos).

Splits are by lesion and frozen across data versions. Patients can still span splits where
lesions share a patient: see limitation 1.

**Classification**, on the pooled client test split (3,262 images, 1,857 lesions). Mean and
range over 3 seeds; the ensemble interval is a 95% bootstrap by lesion. Source:
`results/phase2/v2/test/final_test.json`.

| Model | Balanced accuracy | Melanoma sensitivity | Macro AUROC |
| --- | --- | --- | --- |
| EfficientNet-B0, centralized, single model | 62.99 (62.21–64.28) | 68.50 (66.44–70.12) | 89.68 (88.99–90.43) |
| Ensemble of the 3 centralized seeds (shipped) | 66.46 [60.24, 72.85] | 71.62 [64.21, 78.14] | 90.90 [87.29, 94.62] |
| FedAvg (Flower), single model | 57.14 (56.41–57.81) | 48.36 (46.41–52.25) | |
| FedProx (Flower), single model | 58.98 (57.54–60.12) | 50.86 (49.75–52.09) | |

Federated rounds were selected on pooled validation, which a real federated deployment could
not see, so the federated rows are if anything optimistic.

**At the new hospitals the model degrades sharply.** Shipped ensemble; seven-class accuracy
counts only classes that map cleanly to HAM10000's. Sources:
`results/phase2/analysis_look/results.json`, `results/phase2/triage/test.json`.

| Site | Balanced accuracy | Melanoma sensitivity, top class | Melanoma sensitivity at the triage threshold |
| --- | --- | --- | --- |
| Client test, pooled | 66.46% | 71.62% | 96.2% [93.6, 98.2] |
| Buenos Aires | 55.91% [52.3, 59.6] | 46.64% [39.3, 53.7] | 81.8% [76.7, 86.3] |
| PAD-UFES-20 | 39.40% [36.3, 42.3] | 25.00% [14.7, 36.1] | 78.8% [65.5, 89.5] |

About the triage threshold:
- It is 0.0638 on melanoma probability, set on validation for 95% sensitivity and committed
  before the test look.
- It holds on the client test split and loses 14 to 17 points at the new hospitals. This is the
  silent degradation a drift monitor exists to catch.
- It is a triage framing, not a clinical claim.

**Drift detection**, as AUROC % for telling client test images from each shift. Centralized
models; mean and range over 3 seeds. Source: `results/phase2/drift/v2/table.md`, which also has
FPR@95TPR and the federated models.

| Detector | Buenos Aires | PAD-UFES-20 | PathMNIST | CIFAR-10 |
| --- | --- | --- | --- | --- |
| Mahalanobis (shipped) | 67.8 (66.4–69.5) | 82.3 (79.4–84.0) | 94.5 (92.8–97.2) | 96.4 (93.4–98.5) |
| Gram | 72.6 (71.4–73.3) | 77.2 (75.0–78.4) | 98.5 (96.6–99.5) | 97.6 (95.9–98.9) |
| Max softmax probability | 54.4 (54.0–54.6) | 66.8 (63.0–69.2) | 79.6 (71.5–87.2) | 81.9 (68.9–90.1) |
| Energy | 60.3 (59.3–61.0) | 77.2 (75.4–78.3) | 87.8 (81.7–96.7) | 92.3 (83.9–98.2) |

About the detectors:
- **Mahalanobis was shipped by a rule fixed before any result:** best mean AUROC on the two real
  held-out sites. It won over Gram by 0.18 points, within the seed spread, so Gram runs as a
  backup in Phase 3.
- **The real-site shift is far harder than the benchmark sets.** PathMNIST (histopathology) and
  CIFAR-10 (natural photos) only show that a detector works at all.

### Limitations, measured

1. **Splits are by lesion, not by patient.** At MSK, 47 patients have lesions in two splits.
   HAM10000 and Barcelona record no patient ID, so they cannot be checked. Dropping the affected
   test images moves the ensemble's balanced accuracy from 66.46% to 66.09%.
   [silent-failures #9](docs/silent-failures.md), `results/phase2/patient_overlap/patient_overlap.json`.
2. **Detector scores do not reproduce between GPU and CPU.** Gram's CPU refit missed the GPU
   fit's normalizer by up to 11.8%, and Mahalanobis scores also differ. Monitoring baselines must therefore
   be built on the serving hardware (Phase 3). See [docs/phase2-note.md](docs/phase2-note.md).
3. **Darker skin cannot be judged.** Skin type is recorded only at the two held-out sites, with
   98 images of types IV to VI in total. Within-site results for types I to III are in the
   Phase 2 note, with no skin-type effect claimed.
4. **Site and imaging type are confounded.** PAD-UFES-20 is all smartphone photos, and Buenos
   Aires is mixed, so part of the drop may come from imaging type rather than the hospital.

## Research data: OMOP tables (Phase A)

The lakehouse also publishes **OMOP CDM v5.4-style tables** from silver:
- person, observation_period, condition_occurrence, procedure_occurrence;
- image_occurrence, following the OHDSI medical imaging extension proposal;
- cdm_source and concept.

Seven quality gates fail the pipeline update on any violation: keys, foreign keys, every
source value mapped, concepts resolve, row counts, one label per lesion, required columns.
A failed update stops the job before gold, but the `omop_*` views do not wait for the gate:
they can refresh in the same update. So this is fail-loud, not write-audit-publish.
Every source value maps to an OMOP concept ID looked up in an Athena download, never from
memory, or to concept 0 with a stated reason. SNOMED codes and names stay inside the
workspace; the repository holds concept IDs only.

| Table | Rows |
| --- | ---: |
| person | 15,196 |
| condition_occurrence | 16,238 |
| procedure_occurrence | 37,514 |
| image_occurrence | 27,656 |

Build `0239a19`, pipeline update `5992b755`, all seven gates passed:
`results/research/a_omop/build.json`. Concepts: `results/research/a_omop/concept_check.json`.
Column-by-column definitions, null rates and caveats are in
[docs/data_dictionary.md](docs/data_dictionary.md). Unity Catalog lineage runs from the bronze
tables through `silver_metadata` and `silver_decoded` into every OMOP table:
`results/research/a_omop/lineage.json`, read from `system.access.table_lineage`.

**Caveats**
- Dates are placeholders, because the sources record none.
- A person is a real patient only where ISIC records a patient ID: 2,776 people at MSK
  collection 289, Buenos Aires and PAD-UFES-20. Elsewhere, one person is one lesion.
- Sex is mapped to OMOP Gender concepts. 746 people have no sex recorded and are concept 0.
- Modality is concept 0, because the Athena downloads offered no DICOM vocabulary.

Full Phase 2 record: [docs/phase2-note.md](docs/phase2-note.md). Data engineering:
[docs/phase1-note.md](docs/phase1-note.md).

## Research Phase B: BiomedCLIP baseline

Does a medical vision-language model, frozen, match the shipped EfficientNet ensemble? Not in
balanced accuracy, on any row (one melanoma-sensitivity exception below). BiomedCLIP (pinned
revision `9f341de`) embedded every image once on a Databricks serverless GPU (job run
1040483728228282, 186 seconds of embedding:
[`embeddings_summary.json`](results/research/b_vlm/embeddings_summary.json)). Its prompt template, the
linear probe's settings and the Mahalanobis PCA size were then chosen on train and validation
only, and committed in [`spec.json`](results/research/b_vlm/spec.json) before one test look.
That look ran as a Databricks serverless CPU job (run 930314930675108) and wrote
[`results.json`](results/research/b_vlm/results.json), then its lock
[`test_look.json`](results/research/b_vlm/test_look.json). EfficientNet's numbers come from
its committed analysis-look predictions: no new look. Intervals are 95% bootstrap intervals
(1,000 resamples by patient where ISIC records one, else by lesion, else by image). Resampling by
patient makes the ensemble's interval differ slightly from the by-lesion one in the Phase 2 table.

**Clients' test split** (3,262 images):

| Model | Balanced accuracy | Melanoma sensitivity | Macro AUROC |
|---|---|---|---|
| BiomedCLIP zero-shot | 30.09% (24.95 to 34.70) | 42.74% (36.96 to 48.69) | 0.694 |
| BiomedCLIP linear probe, seed 0 | 50.96% (45.79 to 57.33) | 52.09% (43.71 to 59.80) | 0.838 |
| BiomedCLIP linear probe, mean of 3 seeds | 50.84% (seeds 50.33 to 51.24) | 54.48% (seeds 52.09 to 55.76) | 0.835 to 0.843 |
| EfficientNet-B0 v2 ensemble (shipped) | 66.46% (60.80 to 72.79) | 71.62% (63.52 to 79.01) | 0.909 |

Parentheses are 95% bootstrap intervals, except the seed ranges in the mean row.

- **Paired on the same resamples**, each probe seed is 15.2 to 16.1 points below the ensemble
  in balanced accuracy, and every interval excludes zero (seed 0: -15.50, -20.37 to -11.12).
  Zero-shot is 36.38 points below (-45.99 to -28.05).
- **Without the 47 MSK overlap patients** (3,198 images; all 64 of their test images, not only
  the 55 that share a patient with train as in the Phase 2 limitation): probe 51.00% mean,
  ensemble 66.07%.
- **New hospitals.** Buenos Aires: zero-shot 35.70%, probe 46.58% mean, ensemble 55.91%.
  PAD-UFES-20 (smartphone photos): zero-shot 27.22%, probe 23.40% mean, ensemble 39.40%. Here
  the probe falls below zero-shot.
- **The one exception:** on PAD-UFES-20, zero-shot melanoma sensitivity is higher than the
  ensemble's (71.15% against 25.00%; paired +46.15 points, 31.25 to 60.66), at 27.22% balanced
  accuracy.

**Drift on BiomedCLIP features** (AUROC, the clients' test split against each set; higher
means the detector separates them):

| Detector | Buenos Aires | PAD-UFES-20 | PathMNIST | CIFAR-10 |
|---|---|---|---|---|
| Mahalanobis (PCA 70, method D) | 0.752 | 0.959 | above 0.9999 | above 0.9999 |
| Energy, zero-shot logits | 0.610 | 0.843 | 0.9999 | 0.9998 |
| Energy, probe (mean of 3 seeds) | 0.443 | 0.464 | 0.320 | 0.246 |
| Max softmax, zero-shot | 0.379 | 0.302 | 0.280 | 0.234 |
| Max softmax, probe (mean of 3 seeds) | 0.450 | 0.427 | 0.288 | 0.174 |

- **Max softmax points the wrong way**, and this was checked rather than assumed. Its score
  is the negative top probability, and a CI test checks its direction. The out-of-distribution
  images get a *higher* top softmax probability than the test images. That zero-shot energy
  separates them is consistent with their matching every class text less closely, a
  difference softmax normalizes away. The probe's logits fail both ways, which fits a linear
  head that grows more confident far from its training data. On BiomedCLIP, only Mahalanobis,
  and zero-shot energy on the far-OOD sets, separate the sets well.
- Gram does not apply: it needs convolutional feature maps, and BiomedCLIP is a ViT.

**Caveats.**
- BiomedCLIP was pretrained on PMC-15M figure-caption pairs, which may include figures of ISIC
  or HAM10000 images, so "zero-shot" cannot be shown to be on unseen images.
- The probe is linear on frozen features; fine-tuning was out of scope.
- Research use only, per the BiomedCLIP model card; never served.
- The spec was chosen on the owner's Windows computer and the look ran on Linux. A dry run of
  the whole look on validation gave identical classification numbers on both, and drift AUROCs
  within 0.000001 ([`dry_run/`](results/research/b_vlm/dry_run/)).

## Research Phase C: subgroups and bias

Performance and drift by site, Fitzpatrick skin type, age band, sex and body site, for the
shipped ensemble and BiomedCLIP (linear probe and zero-shot). Sex and body site come from the
OMOP tables as concept IDs (no vocabulary names); age and skin type from silver.

- **Specs:** [`spec.json`](results/research/c_subgroups/spec.json), then
  [`spec_v2.json`](results/research/c_subgroups/spec_v2.json). v2 was committed after the
  first run and a fresh review, to add the MSK-overlap row, zero-shot, skin type, site and
  alert rates.
- **Results:** [`results_v2.json`](results/research/c_subgroups/results_v2.json), Databricks
  serverless CPU job run 534810524836120. Every group the first run
  ([`results.json`](results/research/c_subgroups/results.json)) reported is identical in v2.
- **Two gates, both passed:**
  - before any subgroup, each model's regenerated predictions had to match Phase B's overall
    numbers to 1e-9;
  - before any drift result, the saved per-image drift scores, which carry no IDs, had to line
    up with the images row by row.
- **Reporting rules:** groups under 100 images get counts only, and melanoma sensitivity needs
  100 melanomas. Intervals are 95% bootstrap within each group, for the metrics a group
  reports. Probe results are the mean and range over 3 seeds.

**By site, on the clients' test split** (balanced accuracy; ensemble: 95% bootstrap interval;
probe: range over 3 seeds):

| Client | Images | Ensemble | Probe | Zero-shot |
|---|---|---|---|---|
| HAM10000 Vienna | 1,172 | 67.9% (62.8 to 82.0) | 56.8% (56.3 to 57.8) | 38.1% |
| HAM10000 Queensland | 356 | 69.5% (56.3 to 78.8) | 43.4% (43.3 to 43.6) | 32.0% |
| MSK | 460 | 68.9% (61.3 to 74.8) | 44.6% (43.5 to 45.1) | 20.4% |
| Barcelona | 1,274 | 53.6% (45.1 to 67.5) | 38.0% (37.6 to 38.7) | 30.6% |

Barcelona is the hardest training hospital for the ensemble and the probe (zero-shot does worst
on MSK). Its melanoma sensitivity is still 71.4% (58.5 to 81.0), across 350 melanomas.

**By Fitzpatrick skin type, within each new hospital** (skin type is recorded nowhere else):

| Site | Type | Images | Ensemble | Probe | Drift AUROC, EfficientNet / BiomedCLIP |
|---|---|---|---|---|---|
| Buenos Aires | I | 106 | 56.7% (41.9 to 70.5) | 42.1% | 0.687 / 0.732 |
| Buenos Aires | II | 945 | 52.8% (48.5 to 56.7) | 45.7% | 0.736 / 0.754 |
| Buenos Aires | III | 269 | 61.3% (49.5 to 72.5) | 52.3% | 0.510 / 0.740 |
| PAD-UFES-20 | I | 137 | 39.4% (18.5 to 54.0) | 35.1% | 0.814 / 0.961 |
| PAD-UFES-20 | II | 750 | 43.7% (37.2 to 50.2) | 25.6% | 0.815 / 0.959 |
| PAD-UFES-20 | III | 347 | 43.9% (32.5 to 58.4) | 30.4% | 0.811 / 0.960 |

- Types IV to VI have 25 images at Buenos Aires and 68 at PAD-UFES-20: counts only. **Darker
  skin cannot be judged with this data.**
- Within each site, every interval overlaps, so no skin-type effect is claimed for types I to
  III. Drift uses all images, labelled or not (114 rather than 106 for type I at Buenos Aires).
  Images with no skin type recorded (113 at Buenos Aires, 804 at PAD-UFES-20) have results in
  the file but are not in this table.
- EfficientNet's drift numbers at Buenos Aires reproduce Phase 2's committed skin-type file
  exactly, an independent check that scores were matched to the right images.

**By age band, on the clients' test split** (balanced accuracy; ensemble: 95% bootstrap
interval; probe: range over 3 seeds):

| Age | Images | Ensemble | Probe |
|---|---|---|---|
| 0-29 | 194 | 27.2% (19.3 to 50.6) | 24.9% (24.6 to 25.3) |
| 30-44 | 660 | 60.6% (49.0 to 76.9) | 57.4% (56.5 to 58.6) |
| 45-59 | 961 | 71.8% (64.2 to 79.9) | 50.8% (50.1 to 51.7) |
| 60-74 | 824 | 72.3% (59.5 to 79.6) | 56.8% (56.2 to 57.2) |
| 75+ | 561 | 69.5% (55.0 to 76.4) | 52.4% (51.7 to 52.7) |
| unknown | 62 | counts only | counts only |

- **Under 30 is the weakest group for the ensemble and the probe** (zero-shot does its best
  there, 46.6%). It is mostly nevi (149 of 194 images), and
  the ensemble misses all 6 benign keratoses and all 10 vascular lesions there. It has 29
  melanomas, under the minimum, so its melanoma sensitivity is not reported.
- Balanced accuracy averages only the classes present, and here the rare classes drop in and
  out of resamples, which makes the interval wide and lopsided.
- **Without the 47 MSK overlap patients** (PLAN's rule for every test result): most groups
  barely move, but 75+ drops from 69.5% to 64.4% (52.9 to 73.1) for the ensemble and from
  52.4% to 44.2% for the probe, after removing only 7 of its 561 images. One rare-class image
  can move balanced accuracy this much in a small group. MSK itself goes from 68.9% to 68.3%
  (59.8 to 75.0).

**Sex** (OMOP concepts 8507 and 8532):
- Ensemble balanced accuracy: 63.5% (56.9 to 75.2) and 68.8% (61.4 to 75.7).
- Melanoma sensitivity: 75.3% (65.5 to 83.3) and 67.4% (56.0 to 77.9).
- Both pairs of intervals overlap.

**Body site:** 9 of the 28 body-site groups in the client test split (27 concepts and
"unknown") reach 100 images. The ensemble ranges from 49.1% (concept 4237609, 259 images) to
80.7% (concept 4280364, 421 images). Body site is mixed up with source and class mix, so no gap
is attributed to the site.

**Would a drift alert fire unevenly?** The alert threshold is set at the 95th percentile of the
whole client test split's Mahalanobis scores, so 5% of those images fire overall.
- **EfficientNet (mean of 3 seeds):** Barcelona images fire 7.1%, MSK 2.6%. Age bands range
  from 3.3% (0-29) to 5.9% (30-44), and sex from 4.7% to 5.1%.
- **BiomedCLIP features:** much more uneven by site: MSK 13.3%, Barcelona 6.0%, HAM10000
  Vienna 1.9% and Queensland 1.4%.
- An alert rate that differs by group is a fairness question for the Phase 3 monitor. It is
  reported here, not corrected.

**Drift by subgroup at the new hospitals** (Mahalanobis AUROC, EfficientNet mean of 3 seeds):
- At Buenos Aires, drift rises with age: 0.543 (30-44) to 0.786 (75+) for EfficientNet, and
  0.706 to 0.818 on BiomedCLIP features. The two sex groups differ by 0.008 for EfficientNet
  and 0.013 on BiomedCLIP features.
- At PAD-UFES-20, every group with at least 100 images is far from the training data: 0.790 to
  0.883 for EfficientNet and 0.931 to 0.972 on BiomedCLIP features. That range includes the
  804 images with no sex recorded.
- These are descriptive. Age at Buenos Aires also changes the class mix and the lesions
  imaged, so this does not show that age itself causes drift.

## Research Phase D: Grad-CAM review worksheet and error analysis

A worksheet for a human reviewer, not a finding. Under
[`spec.json`](results/research/d_review/spec.json) (committed first), one Databricks serverless
CPU run (624234357242388) picked 150 client-test images from the committed predictions (two
vascular lesions are on both lists):
- the shipped ensemble's worst misses: up to 20 per true class, fewer where a class has fewer
  misses (17 dermatofibromas, 15 vascular lesions);
- the 20 most out-of-distribution images by Mahalanobis.

It drew Grad-CAM maps from the shipped seed-0 checkpoint, for the predicted and the true class.
Before any map, seed 0 had to reproduce its committed probabilities on these images; the
largest gap was 0.0000002.

**Sanity checks** ([`results.json`](results/research/d_review/results.json),
[`worksheet.csv`](results/research/d_review/worksheet.csv), IDs and values only):
- **Weight randomization** (Adebayo et al.): the mean rank correlation between each map and the
  same map from a re-initialized network is 0.045 (range -0.63 to 0.64). So for most images
  the maps depend on what the network learned. That is a minimum bar: it does not show the maps
  are faithful to how the model decides. 11 maps correlate 0.5 or more and are flagged
  "not interpreted".
- **Agreement across seeds:** seed 0's maps correlate 0.81 on average with seeds 1 and 2
  (range -0.07 to 1.00). Some images get different explanations from equally trained models.
- The maps are 7x7 upsampled to 224 px: coarse. The smooth upsampling likely raises these
  pixel-level correlations, seed agreement most.

The worksheet is an MLflow artifact (run `885408cb6a9d4f5f8b38ea53598cdfe8`, with each image's
license and attribution) and the table `workspace.cdm.gold_review_worksheet`, whose
`reviewer_notes` column is blank. Nothing is reviewed until a person fills it in.

**Error analysis** (the Phase 2 item moved here): the ensemble's confusion matrices per client
and per skin type at the new hospitals are in `results.json`. On the clients' test split, the
largest confusions are nevi called melanoma (199 images), melanomas called nevi (86) and
benign keratoses called melanoma (68).

## Research Phase E: image plus metadata

Does adding age, sex and body site to a frozen image embedding help? The same linear head
(`cdm.probe`) was fit on the same frozen embeddings, with and without metadata. Metadata here
means age, missing-value flags, and sex and body site as OMOP concept IDs; the encoding is fit on
train only, and the image's source is never an input.

Two embedding sets were used:
- BiomedCLIP (Phase B);
- ImageNet-pretrained EfficientNet-B0 from `gold_embeddings`. These are not the shipped model's
  own features.

Settings were chosen per variant on validation and committed as
[`spec.json`](results/research/e_fusion/spec.json) (selection job run 230673257459446). Then one
test look: Databricks serverless CPU job run 75228269324996,
[`results.json`](results/research/e_fusion/results.json). Its attempt log shows one attempt.

**Balanced accuracy, mean of 3 seeds (range):**

| Variant | Client test | Without MSK overlap | Buenos Aires | PAD-UFES-20 |
|---|---|---|---|---|
| Metadata only | 36.4% (35.7 to 36.8) | 36.5% | 38.5% | 31.7% |
| BiomedCLIP, image only | 50.8% (50.3 to 51.2) | 51.0% | 46.6% | 23.4% |
| BiomedCLIP, image + metadata | 53.8% (53.2 to 54.1) | 54.4% | 52.2% | 26.4% |
| ImageNet B0, image only | 54.1% (53.7 to 54.5) | 53.6% | 44.2% | 27.2% |
| ImageNet B0, image + metadata | 56.4% (55.8 to 57.3) | 56.3% | 45.8% | 29.1% |
| Shipped ensemble (reference) | 66.46% | 66.07% | 55.91% | 39.40% |

**Paired differences, image + metadata minus image only (same seed, same resamples):**
- **Client test:** +2.8 to +3.1 points for BiomedCLIP and +1.9 to +2.8 for ImageNet B0. Four of
  the six 95% intervals include zero.
- **Buenos Aires:** +5.0 to +6.6 for BiomedCLIP, every interval clear of zero. ImageNet B0 gains
  +0.8 to +2.4, with intervals spanning zero.
- **PAD-UFES-20:** +2.4 to +4.2 for BiomedCLIP and +1.7 to +2.1 for ImageNet B0 (one of
  the six intervals includes zero).
- **HAM10000 test images alone:** all six intervals include zero.

**Shortcut checks:**
- **Metadata alone carries signal:** 36.4% against 14.3% for chance over seven classes. Part of
  the gain may be prevalence by age and body site rather than anything about the lesion, and
  metadata, including which fields are missing, can encode the source.
- **At PAD-UFES-20, metadata alone (31.7%) beats every image and fusion variant (23.4% to
  29.1%):** the clearest sign in this table that metadata works partly as a shortcut there.
- **Source held out:** trained without one hospital and scored on that hospital's test images.
  - With BiomedCLIP, metadata helped at all three: Barcelona 26.9% to 29.9%, HAM10000 45.0% to
    46.9%, MSK 33.2% to 37.9%.
  - With ImageNet B0 it helped at Barcelona (28.3% to 31.7%), barely moved HAM10000 (44.3%
    to 44.9%) and hurt at MSK (49.7% to 41.6%).
  - These are point estimates over 3 seeds, without intervals, and they reuse the settings
    chosen with every source in train and validation.
- **Without the MSK overlap patients**, the gains are the same size or larger, so they are not
  explained by those 47 patients. Patient overlap elsewhere cannot be checked (HAM10000 and
  Barcelona record no patient ID).

**Result:** metadata adds a few points to a frozen embedding, most clearly for BiomedCLIP at
Buenos Aires. No variant comes near the fine-tuned ensemble.

**Follow-up: the shipped model's own features** (owner's decision, option b). One GPU job
(run 498023462798649, 154 seconds of extraction) took the penultimate features of each shipped
seed. A gate checked the test rows against the committed predictions; the mean gap in top
probability was 0.0058 to 0.0065.

Two new variants were fit, image only and image plus metadata. Probe seed s trains on
checkpoint s's features. Settings were chosen on validation
([`spec_shipped.json`](results/research/e_fusion/spec_shipped.json)), then the pair had its own
single test look (run 124072636853443,
[`shipped/results.json`](results/research/e_fusion/shipped/results.json)).

| Variant (shipped features) | Client test | Buenos Aires | PAD-UFES-20 |
|---|---|---|---|
| Image only | 62.5% (62.0 to 63.1) | 49.7% (48.1 to 51.2) | 37.6% (35.5 to 39.3) |
| Image + metadata | 63.0% (62.6 to 63.5) | 50.7% (48.5 to 52.2) | 38.2% (37.6 to 38.9) |

- **With the model's own features, metadata adds almost nothing.** The paired gain is +0.2 to
  +0.8 points on the client test split, and all three intervals include zero.
- At the new hospitals, 2 of the 3 intervals at each site include zero.
- The gains on the frozen generic embeddings above shrink to this once the image
  representation is strong.
- **Caveat:** these features come from networks trained on the train split, so the head is
  fit on images the network already fits. Both variants also chose the grid's lowest learning
  rate.

**Known limits:**
- Both ImageNet-feature variants chose the grid's lowest learning rate, and the grid was not
  widened.
- PLAN asked for the shipped EfficientNet's own penultimate features, "already in
  `gold_embeddings`". That table holds ImageNet-pretrained B0 features (the Phase 1 job wrote
  them; recomputing with the shipped model is a Phase 3 item), so those were used. The owner
  chose to keep them and add the shipped model's own features (the follow-up above).
- The committed job definition says `mode: select`; the look ran with the run-time
  override `mode=look`, which Databricks records for run 75228269324996.

## Research informatics services (Phase G)

A small research data service on top of the OMOP tables. It is a portfolio build on public
data and **synthetic** requests, not a clinical service.

- **Requests** come in through an issue form and are tracked on the
  [Research data requests](https://github.com/users/erjonb19/projects/2) board. Seven
  synthetic requests (#8 to #14, labelled `synthetic`) went through the whole path.
- **Governed cohort service** (`src/cdm/cohort.py`, `phenotypes/`): versioned YAML phenotypes
  in an allowlisted language; a definition runs only after a reviewer approves its exact hash;
  every refusal, decision and run goes to a hash-chained audit log. Counts under 11 are
  suppressed, as are counts that subtraction would reveal, both within a result and against
  earlier results for nested definitions (`docs/silent-failures.md` 10). It runs on DuckDB and
  on Databricks SQL.
- **Service metrics:** [docs/service_metrics.md](docs/service_metrics.md), generated from the
  issue tracker, the audit log and committed results.
- **FAIR packaging:** `metadata/dataset.jsonld`, `CITATION.cff`, and an investigator
  quickstart notebook (`notebooks/quickstart.ipynb`) that runs on public PAD-UFES-20 metadata.
- **Protocol and governance:** [docs/EVAL_PLAN.md](docs/EVAL_PLAN.md) and
  [docs/governance.md](docs/governance.md).
- **Public demo** (`app/cohort_demo.py`, logic in `src/cdm/demo.py`): the cohort service on
  public PAD-UFES-20 metadata, counts only, concept IDs only. Live at
  [cdm-cohort-demo.streamlit.app](https://cdm-cohort-demo.streamlit.app/) (Streamlit Community Cloud, deployed 2026-10-22). Walkthrough script: [docs/demo_walkthrough.md](docs/demo_walkthrough.md).
- **Clinical text extraction** (`src/cdm/extract.py`, `scripts/extraction.py`): Claude Haiku
  reads a note and returns diagnosis, body site and histopathology confirmation as Phase A
  concept IDs, against a keyword baseline. It is built and tested on made-up notes only. It has
  not been scored: that needs MTSamples notes and a labelled set, so there are no extraction
  results yet. Note text never enters the repository.

## Phase 0 results (HAM10000 only, CPU, history)

Kept as the record of fixing the course notebook's science. These numbers are HAM10000-only
and not comparable with Phase 2's four-hospital test split.

EfficientNet-B0 fine-tuned on HAM10000 with a lesion-level split (no lesion in two
splits), 12 epochs, class-weighted loss, trained on CPU. The lesion split is fixed; three
training seeds. Each cell is the mean over seeds with the min–max range, in percent.
Source: [`results/phase0/results.md`](results/phase0/results.md), produced by
`python -m cdm.reproduce` from code `2ccda00`.

Balanced accuracy on the HAM10000 test split (1,528 images): **78.0 (76.4–79.3)**

| Class | Test images | Recall |
| --- | ---: | ---: |
| akiec | 53 | 68.6 (66.0–69.8) |
| bcc | 88 | 81.4 (78.4–84.1) |
| bkl | 164 | 69.9 (65.2–76.8) |
| df | 21 | 84.1 (71.4–90.5) |
| mel | 162 | 69.3 (61.7–77.2) |
| nv | 1,022 | 87.5 (86.5–88.6) |
| vasc | 18 | 85.2 (83.3–88.9) |

df and vasc have 21 and 18 test images, so their recall is noisy.

Out-of-distribution detection. In-distribution is the HAM10000 test split; each OOD set
is a fixed 2,000-image subset. AUROC: higher is better. FPR@95TPR: share of OOD images
accepted when 95% of in-distribution images are kept; lower is better.

| Detector | PathMNIST AUROC | PathMNIST FPR@95TPR | CIFAR-10 AUROC | CIFAR-10 FPR@95TPR |
| --- | ---: | ---: | ---: | ---: |
| Mahalanobis (PCA 256) | 94.6 (93.8–95.0) | 31.4 (26.2–34.3) | 84.7 (80.1–92.6) | 62.8 (49.5–70.2) |
| Max softmax probability | 84.2 (82.4–86.5) | 62.7 (56.0–67.6) | 87.5 (81.4–92.5) | 47.5 (35.0–58.0) |
| Energy | 90.4 (83.4–95.6) | 36.3 (24.3–48.4) | 93.6 (87.7–96.8) | 19.4 (10.9–27.7) |

PathMNIST (colon histopathology) and CIFAR-10 (natural photos) are known, clearly
different image types. They test whether a detector works at all, not how it handles a
real shift between hospitals; that comes in Phase 2. No drift in this table is staged:
the OOD sets are real, separate datasets.

### Read these numbers with three caveats

1. **CIFAR-10 Mahalanobis is unstable.** Across seeds it spans 80.1% to 92.6%, and a rerun
   of seed 1 gave 95.7%. Fine-tuning pulls CIFAR-10 images onto the majority class
   (common moles), which makes this detector fragile on them.
2. **Mahalanobis depends heavily on the PCA size.** On one fine-tuned model, CIFAR-10
   AUROC goes from 82.6% to 99.2% depending on the number of components. 256 is reported
   because it was fixed before any result was seen; picking the best-scoring size would
   be tuning to the test set.
3. **The ranges understate run-to-run variation.** The reported seeds were seeded after
   the classifier head was initialised, so a rerun of seed 1 did not reproduce it
   (fixed since, with a test; not rerun). One rerun moved some numbers further than the
   whole seed range:

   | Metric | Seed spread (3 seeds) | Seed 1 rerun change |
   | --- | ---: | ---: |
   | Balanced accuracy | 2.9 | 4.6 |
   | PathMNIST Mahalanobis AUROC | 1.2 | 4.2 |
   | PathMNIST max softmax AUROC | 4.1 | 0.9 |
   | PathMNIST energy AUROC | 12.2 | 0.4 |
   | CIFAR-10 Mahalanobis AUROC | 12.5 | 15.6 |
   | CIFAR-10 max softmax AUROC | 11.2 | 0.5 |
   | CIFAR-10 energy AUROC | 9.1 | 0.8 |

   Percentage points. Seed spread is max minus min over the three seeds.

Details and all supporting numbers: [docs/phase0-note.md](docs/phase0-note.md).

### What the ECE 570 notebook got wrong

The course notebook reported a Mahalanobis AUROC of 4.1% for HAM10000 vs CIFAR-10. It
scored in-distribution images that its PCA had been fit on, used each image's true class
for in-distribution but the nearest class for OOD, and projected PathMNIST with a PCA refit
on itself. [docs/phase0-note.md](docs/phase0-note.md) traces each fault to a notebook cell,
and each now has a test.

## Reproduce

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train]"
CDM_DATA=/path/to/data python -m cdm.reproduce   # downloads and checksums all data
```

Measured: 2.6 to 4.4 hours per seed on a 14-core laptop CPU, depending on load. Data comes
from the sources in [docs/DATA_CARD.md](docs/DATA_CARD.md) and is never committed. The table
above came from code `2ccda00`, which seeded after building the model. Current code seeds
first and runs deterministically, so a rerun is exactly repeatable but gives numbers within
the variation shown in caveat 3, not these exact numbers.

## Honesty docs

- [docs/BUILT_VS_PLANNED.md](docs/BUILT_VS_PLANNED.md): built, scaffolded, not started.
- [docs/silent-failures.md](docs/silent-failures.md): bugs that reported success while
  wrong, each with the check that now catches it.
- [docs/DATA_CARD.md](docs/DATA_CARD.md): sources, licenses, checksums.

The original Purdue ECE 570 notebook and slides are kept in [reference/](reference/) as
read-only history.

## License

The code in this repository is released under the [MIT License](LICENSE). MIT covers the
code only. ISIC images and metadata, and every other source dataset, keep their own
licenses (CC-0, CC-BY or CC-BY-NC per ISIC image, and the terms listed per source in
[docs/DATA_CARD.md](docs/DATA_CARD.md)). The course notebook and slides in
[reference/](reference/) are not covered by the MIT License either; third-party figures and
images in them keep their original terms.

What the repository holds besides code, all derived from the sources and keeping their
terms (non-commercial where a source is CC BY-NC):
- `config/splits_v1.csv` and `config/patient_overlap_v1.csv`: ISIC lesion and patient IDs
  with their split;
- `config/omop_concepts.csv`: source values mapped to OMOP concept IDs (no SNOMED text);
- `results/phase2/analysis_look/predictions.csv.gz`: one row per test image, with ISIC IDs,
  labels and each model's class probabilities;
- a few HAM10000 sample images in one cell output of the notebook in reference/ (CC BY-NC
  4.0, Tschandl et al. 2018);
- aggregate results (JSON).

No image files, dataset downloads or model weights are committed.
