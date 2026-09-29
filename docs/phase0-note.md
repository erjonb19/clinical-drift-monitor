# Why the notebook's 4.1% AUROC was wrong

The ECE 570 notebook (`reference/ECE570_Project.ipynb`) reported a Mahalanobis AUROC of
**4.1%** for HAM10000 vs CIFAR-10 and **43.4%** for HAM10000 vs PathMNIST. Both are below
the 50% a coin flip would get, which means the score ranked CIFAR-10 and PathMNIST as *more*
like skin lesions than skin lesions. That is a broken comparison, not a weak detector. Every
fact below is from the notebook's own code and printed output.

## HAM10000 vs CIFAR-10: 4.1%

**1. In-distribution was scored on the images the statistics were fit on.** Cell 6 fits a
300-component *whitened* PCA on 1,002 HAM10000 images, and cell 7 scores those same 1,002
images. Their features are 62,720-dimensional (the unpooled 1280 × 7 × 7 feature map), so
300 components fit to 1,002 points mostly describe those particular points. Whitening
scales every component to unit variance over them, so each fitted point lands about
√300 ≈ 17.3 from the centre by construction. The notebook printed an in-distribution mean
of **16.49**.

An image the PCA has not seen, skin lesion or not, has little energy in those
sample-specific directions and projects much closer to the centre. CIFAR-10 printed a mean
of **6.81**. So "seen during fitting" was being compared with "not seen", and the score
ranked unseen images as in-distribution.

**2. The two sides were scored against different targets.** In-distribution images were
scored against the mean of their *true* class (cell 7), CIFAR-10 against the *nearest* class
mean (cell 11). The nearest mean is never farther than the true one, which pushes in the
same wrong direction.

**3. No held-out data existed.** The model trained on a random 10% of *images* for 3
epochs with no validation or test split, so there were no unseen in-distribution images to
score, and no accuracy to show the model had learned anything.

## HAM10000 vs PathMNIST: 43.4%

**4. PathMNIST lived in a different space.** Cell 16 loads a fresh ImageNet EfficientNet,
not the fine-tuned model. Cell 20 refits PCA on 1,000 PathMNIST *training* images. The code
comment says "HAM features", but `train_loader` there is PathMNIST. Cell 21 then measures
those points against class means and a covariance from HAM10000's PCA space. The distances
compare coordinates from two unrelated projections. Both sides were points their own PCA
had been fit on, so both came out near √300 (means 16.49 and 15.80), and the AUROC landed
near chance.

## What Phase 0 does instead

| Notebook | Phase 0 (`src/cdm/`) | Enforced by |
| --- | --- | --- |
| Random image-level 10% sample, no held-out data | Lesion-level 70/15/15 split of all 10,015 images | `tests/test_splits.py` |
| In-distribution scored on the fitting images | Fit on the train split, score the test split | `cdm.reproduce.run_seed` |
| True class for ID, nearest class for OOD | Nearest class mean for every input; `score` takes no labels | `test_mahalanobis_uses_nearest_class_not_true_class` |
| PCA refit on PathMNIST; ImageNet model for PathMNIST | One fine-tuned model and one PCA for every set | `test_mahalanobis_statistics_do_not_change_when_scoring` |
| 62,720-d unpooled features, whitened PCA | 1,280-d pooled features (what the head classifies), shared class-centred covariance | `test_features_feed_the_classifier_head` |
| A single run | 3 training seeds, mean and range | `python -m cdm.reproduce` |
| An AUROC below 50% reported | Any AUROC below 50% exits non-zero | `test_implausible_auroc_is_flagged` |

The corrected numbers are in [`results/phase0/results.md`](../results/phase0/results.md).

## Follow-up: Mahalanobis does worse on CIFAR-10 than on PathMNIST

The Phase 0 run (code `2ccda00`, 3 seeds, PCA 256) gave Mahalanobis an AUROC of
**84.7% (80.1–92.6)** on CIFAR-10 and **94.6% (93.8–95.0)** on PathMNIST. CIFAR-10 is
natural photos and was expected to be the easier set to flag. A result that runs against
expectation gets checked before it is reported, so four checks were run.

The run saved metrics but not model weights ([silent-failures #5](silent-failures.md)), so
seed 1, the worst CIFAR-10 seed, was retrained with checkpoint saving (code `06245f4`,
`results/phase0_seed1_rerun/`). That retrain did not reproduce seed 1 (next section), so the
fine-tuned checks below describe *a* seed 1 model, not the one behind the reported 80.1%.

### Short answer

- The scores point the right way in every seed and every setting checked.
- Fine-tuning pulls unfamiliar images onto the majority class. After fine-tuning, 99% of
  CIFAR-10 images sit nearest common moles (nv). The pretrained features spread them out.
- Mahalanobis results depend heavily on the PCA size, before and after fine-tuning.
- CIFAR-10 Mahalanobis is simply unstable under this training recipe: the same recipe has
  given 80.1% to 95.7%. The 84.7% mean is reported as measured, with its range. It should
  not be read as "CIFAR-10 is harder than PathMNIST" for this detector.

### Reproducibility: the same seed did not give the same model

| Seed 1 | Reported run | Rerun | Reported 3-seed range |
| --- | ---: | ---: | --- |
| Best epoch | 5 | 8 | |
| Balanced accuracy | 76.5% | 81.0% | 76.4–79.3 |
| CIFAR-10 Mahalanobis AUROC | 80.1% | 95.7% | 80.1–92.6 |
| PathMNIST Mahalanobis AUROC | 93.8% | 98.0% | 93.8–95.0 |
| CIFAR-10 energy AUROC | 96.3% | 97.1% | 87.7–96.8 |

Same seed, data, split and settings. The only code change was saving the weights after
training. The two runs already differed at epoch 1 (loss 1.1328 vs 1.1429). The likely
cause is multi-threaded floating-point arithmetic on the CPU, whose summation order can
change between runs, especially under different machine load. This is not yet confirmed.

What it means: a fixed seed does not pin down the model on this machine, and the spread
between two runs of one seed is as large as the spread between seeds. Three seeds
therefore understate the true run-to-run variation. See
[silent-failures #6](silent-failures.md).

### Check 3: score direction per seed — correct

In every seed, every detector scores both OOD sets higher than the HAM10000 test images
(every AUROC above 50%; `results/phase0/seed*.json`). In both checked feature sets, the
median Mahalanobis score of each OOD set is above the in-distribution median at every PCA
size (`results/phase0/checks/*.json`).

### Check 1: pretrained vs fine-tuned features — fine-tuning costs CIFAR-10 the most

Same setup throughout (PCA 256, fit on HAM10000 train only, same test split and OOD subsets):

| Features | PathMNIST AUROC | CIFAR-10 AUROC |
| --- | ---: | ---: |
| ImageNet, before fine-tuning | 97.4 | 98.7 |
| Fine-tuned, reported 3 seeds (mean, range) | 94.6 (93.8–95.0) | 84.7 (80.1–92.6) |
| Fine-tuned, seed 1 rerun | 98.0 | 95.7 |

Every fine-tuned model scores CIFAR-10 below the pretrained features. The size of the loss
varies a lot from run to run, from 3 points to 19 points.

### Check 2: which lesion class OOD images land nearest (PCA 256)

Percent of each set's images whose nearest class mean is that class:

| Features | Set | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | HAM10000 test | 5.8 | 6.8 | 10.8 | 3.3 | 11.6 | 60.5 | 1.2 |
| Pretrained | PathMNIST | 38.4 | 9.9 | 6.9 | 1.9 | 0.2 | 35.0 | 7.6 |
| Pretrained | CIFAR-10 | 3.8 | 8.2 | 49.7 | 10.5 | 4.8 | 15.0 | 8.1 |
| Fine-tuned (seed 1 rerun) | HAM10000 test | 2.5 | 4.0 | 9.6 | 0.7 | 10.0 | 72.2 | 1.0 |
| Fine-tuned (seed 1 rerun) | PathMNIST | 0.0 | 0.2 | 14.6 | 0.2 | 0.2 | 84.5 | 0.2 |
| Fine-tuned (seed 1 rerun) | CIFAR-10 | 0.0 | 0.0 | 0.4 | 0.1 | 0.0 | 99.0 | 0.5 |

After fine-tuning, almost every CIFAR-10 image lands nearest nv, the majority class (67%
of training images). Whether a CIFAR-10 image gets flagged then depends on how far it
sits from that one class, rather than on being far from all of them. That is a plausible
reason the CIFAR-10 result is fragile, but it is shown here on one model only.

### Check 4: PCA size — a key sensitivity

AUROC and FPR@95TPR by number of PCA components:

| PCA components | Pretrained PathMNIST | Pretrained CIFAR-10 | Fine-tuned PathMNIST | Fine-tuned CIFAR-10 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 75.1 / 87.6 | 66.5 / 99.6 | 87.7 / 67.2 | 85.6 / 75.4 |
| 64 | 87.2 / 67.7 | 74.0 / 96.3 | 89.6 / 63.6 | 82.6 / 80.0 |
| 128 | 93.6 / 43.0 | 91.4 / 56.5 | 95.3 / 28.6 | 91.0 / 47.5 |
| **256 (reported)** | **97.4 / 12.2** | **98.7 / 5.9** | **98.0 / 11.3** | **95.7 / 21.3** |
| 512 | 99.1 / 5.9 | 99.9 / 0.0 | 99.1 / 3.1 | 98.0 / 9.8 |
| 1280 (no reduction) | 99.4 / 4.1 | 100.0 / 0.0 | 99.7 / 0.1 | 99.2 / 3.1 |

Each cell is AUROC / FPR@95TPR in percent. "Fine-tuned" is the seed 1 rerun. For both
feature sets, the number of components changes the answer more than the choice of OOD set
does. The signal that separates OOD images sits in the low-variance directions a small
PCA throws away.

**PCA 256 stays the reported setting.** It was fixed in `MahalanobisDetector` before any
result was seen. Switching to the size that scores best on these OOD sets would be tuning
to the test set, and in deployment the OOD data is unknown. A fair way to pick the size
would use validation data only, and belongs to Phase 2, not to this report.
