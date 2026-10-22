# Clinical Imaging Drift Monitor

A skin lesion classifier trained across hospitals, with an out-of-distribution (drift) score
tested on real new hospitals. Built on Databricks Free Edition at zero cost. A personal
portfolio project and research prototype: **not for clinical use**.

**Status (2026-10-21):**
- Phases 0 to 2 are done: fix the science, data engineering, and ML. The one Phase 2 item left,
  error analysis, moved to research Phase D.
- The research extension (Phases A to F: OMOP data, a vision-language baseline, subgroups,
  interpretability, fusion) is planned in
  [docs/research_extension_plan.md](docs/research_extension_plan.md). Phase A (OMOP tables)
  is built; Phase G (research data services) is in progress.
- Phases 3 to 6 have not started.

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
