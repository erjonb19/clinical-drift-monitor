# Clinical Imaging Drift Monitor

A skin lesion classifier served with an out-of-distribution (drift) score. See
[PLAN.md](PLAN.md) for scope and order, and [docs/BUILT_VS_PLANNED.md](docs/BUILT_VS_PLANNED.md)
for what exists today.

**Status:** Phase 0 (fix the science) has results and is awaiting sign-off. Phases 1 to 6
have not started.

## Phase 0 results

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
pip install -e ".[dev]"
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
