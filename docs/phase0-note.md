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

The corrected numbers are in [`results/phase0/results.md`](../results/phase0/results.md)
once the Colab run is committed. Until then, this note makes no claim about what they are.
