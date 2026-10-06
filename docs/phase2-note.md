# Phase 2 note

Working record of Phase 2 decisions and findings. Every number links to committed evidence.
Sections are added as results land.

## Calibration: the v2 ensemble is used uncalibrated

Temperature scaling (one temperature fitted on validation by negative log-likelihood) is
reported for every model. For the v2 3-seed ensemble it made calibration worse on test: on
v1's exact test images, expected calibration error rose from 2.19% to 3.57% (fitted
temperature 1.12), and on the full v2 test set from 2.00% to 3.56%. The averaged
probabilities of three seeds are already close to calibrated, and the temperature fitted on
validation did not carry over to test. The ensemble is therefore used **uncalibrated**.
Single models are still improved by it (v2 centralized, mean of 3 seeds: 5.91% to 2.60% on
v1's test images). Evidence: `results/phase2/v2/test/final_test.json`.

## Centralized vs federated: the gap widened with more Barcelona data

Pooled test balanced accuracy on v1's exact test images (2,623 images), mean of 3 seeds:

| Data | Centralized | FedAvg | FedProx | Gap to FedAvg | Gap to FedProx |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (Barcelona capped at 5,000) | 60.64% | 57.72% | 56.76% | 2.9 points | 3.9 points |
| v2 (Barcelona capped at 10,000) | 64.54% | 58.00% | 59.84% | 6.5 points | 4.7 points |

Adding Barcelona data helped centralized training by 3.9 points but FedAvg by only 0.3
(FedProx by 3.1), so the gap between centralized and federated training widened from about
3 points to about 6 for FedAvg (about 5 for FedProx). Barcelona's share of training images,
which is also its FedAvg aggregation weight, rose from 25% to 39%. Federated rounds were
selected on pooled validation, which a real federated deployment could not see (PLAN.md),
so these federated numbers are, if anything, optimistic. Evidence:
`results/phase2/overnight/final_test.json`, `results/phase2/overnight/rerun-fedprox-seed1/`,
`results/phase2/v2/test/final_test.json`, `results/phase2/v2/training/data_summary.json`.

## Drift detector: Mahalanobis shipped, a near tie with Gram

The ship rule, fixed before any Phase 2 result, ranks detectors by mean AUROC on the two real
held-out sites (Buenos Aires and PAD-UFES-20) for the centralized models, with FPR@95TPR only
breaking ties. Mahalanobis (PCA size by method D) scored 75.05% and Gram
74.87%: a margin of 0.18 points, smaller than the seed-to-seed range (Mahalanobis at
Buenos Aires alone spans 66.35% to 69.46%). The rule picked Mahalanobis and it is kept.
Gram has the lower false-alarm rate: mean FPR@95TPR on the real sites 75.09% against
82.02% for Mahalanobis. The two split the real sites: Gram is better at Buenos Aires,
Mahalanobis at PAD-UFES-20. Gram therefore runs alongside Mahalanobis as a backup detector in
Phase 3 monitoring (PLAN.md). Evidence: `results/phase2/drift/v2/table.json`, `table.md`.

## Limitation: method C's PCA search topped out

Method C (the sensitivity row: the PCA size with the best mean AUROC on benchmark selection
draws, from 16 to 512 components) chose 512, the largest candidate, for every one of the nine
models, so its search range was too small to find a peak. Method D, the primary rule, chose
517 to 631 components. The method C row stays within about 0.5 AUROC points of method D
everywhere, and it never enters the ship rule, so it is not rerun; it is reported as a
limitation. Evidence: `pca_components` in `results/phase2/drift/v2/table.json`.

## Main finding: the triage operating point loses 14 to 17 points of sensitivity at new sites

The triage operating point (triage framing, not a clinical claim) refers an image when the
shipped ensemble's melanoma probability is at or above 0.0638, the highest threshold
reaching 95% melanoma sensitivity on v2 validation (553 melanoma images in
208 lesions, of 3,210 images). The threshold was committed before any test image was
scored, and each model had one triage look. On the clients' test split it holds: melanoma
sensitivity 96.2% [93.6, 98.2]. At the two held-out sites it does not: 81.8%
at Buenos Aires and 78.8% at PAD-UFES-20, drops of 14.3 and 17.3 points. This is
the failure the drift monitor exists to catch: a model whose numbers look right where it was
built, and that quietly misses more melanomas at a new site. Evidence:
`results/phase2/triage/threshold.json`, `results/phase2/triage/test.json`.

| Group | Images | Melanomas (prevalence) | Specificity | Sensitivity | Referral rate |
| --- | ---: | ---: | --- | --- | ---: |
| Client test, pooled | 3,262 | 599 (18.4%) | 52.3% [49.7, 55.2] | 96.2% [93.6, 98.2] | 56.6% |
| Barcelona | 1,274 | 350 (27.5%) | 47.1% [41.3, 52.6] | 95.4% [91.3, 98.3] | 64.6% |
| HAM Queensland | 356 | 54 (15.2%) | 40.7% [34.1, 47.4] | 98.1% [94.0, 100.0] | 65.2% |
| HAM Vienna | 1,172 | 108 (9.2%) | 64.8% [61.2, 68.8] | 96.3% [91.3, 100.0] | 40.8% |
| MSK | 460 | 87 (18.9%) | 39.1% [33.5, 44.4] | 97.7% [93.8, 100.0] | 67.8% |
| Buenos Aires (external) | 1,458 | 253 (17.4%) | 59.1% [56.4, 62.1] | 81.8% [76.7, 86.3] | 48.0% |
| PAD-UFES-20 (external) | 2,106 | 52 (2.5%) | 62.5% [60.2, 64.7] | 78.8% [65.5, 89.5] | 38.6% |

**Specificity is the metric that carries between settings; the referral rate is not.**
Specificity (the share of non-melanoma images not referred) does not depend on how common
melanoma is. The referral rate does: melanoma prevalence is about 18% in these datasets (599
of 3,262 client test images), far above what a primary-care population would see, so the
referral rates above describe these datasets only. At low prevalence the referral rate tends to
1 − specificity (about 48% at the pooled specificity of 52.3%): only somewhat lower than here,
and made almost entirely of false referrals. No comparison is made with any marketed device. As context
for why checks in new populations matter after release: when FDA authorized DermaSensor, an
AI-enabled skin cancer device for primary care, in January 2024, it required further
post-market clinical validation in patients from demographic groups representative of the
U.S. population, including groups with limited melanoma representation in the premarket
studies (FDA Roundup, January 16, 2024).

**Part of the drop at the external sites may come from imaging type, not only the site.** All
images at the four training clients are dermoscopic. PAD-UFES-20 is entirely clinical
close-up photographs taken with smartphones (Pacheco et al., 2020, "PAD-UFES-20: a skin lesion
dataset composed of patient data and clinical images collected from smartphones", Mendeley
Data, doi:10.17632/zr7vgbcyr2.1; all 2,298 images recorded as "clinical: close-up" in ISIC).
Buenos Aires is mixed: 1,270 dermoscopic and 346 clinical images (ISIC `image_type`). With
only 52 melanomas, PAD-UFES-20's interval is also wide. Site and imaging type are not
separated here.

## Finding: Gram did not reproduce between GPU and CPU

The batch-drift reference job (run 828284284699299, CPU) refit each centralized model's
detectors on training images to rebuild validation scores, and had to reproduce the drift
run's (GPU) fit within a tolerance committed in advance. For v2-final-seed0, Mahalanobis's
method-D PCA size reproduced exactly (541 components), but Gram's per-layer normalizer differed
from the GPU fit by up to 11.8%, against a tolerance of 0.1%. The job stopped as specified and
the tolerance was not loosened. Gram was then dropped from batch-level drift (a recorded spec
change, made before any batch-drift result existed).

Hypothesis, not verified: on the A10 GPU, PyTorch's default convolution arithmetic (TF32)
differs slightly from CPU float32; Gram raises activations to powers up to 10, which magnifies
those differences, and then takes per-class minimums and maximums, which are sensitive to
them. Whatever the cause, Gram scores are only comparable to a fit made on the same hardware
and settings, which matters for running Gram as a backup detector (PLAN.md, Phase 3). Evidence:
the failed run's error output, recorded in `results/phase2/batch_drift/spec.json`.

## Label provenance

Every melanoma label at every site is confirmed by histopathology, so melanoma sensitivity
rests on biopsy-proven labels. Specificity rests partly on benign labels confirmed without
biopsy (serial imaging, single-image expert consensus or confocal microscopy), especially
outside MSK: histopathology confirms 95.3% of MSK's nevi but 64.6% of
Barcelona's, 37.3% of HAM10000's and 31.4% of Buenos Aires'. Overall
histopathology shares: MSK 97.7%, Barcelona 75.2%, Buenos Aires
64.1%, PAD-UFES-20 58.4%, HAM10000 53.3%.

**Provenance gap.** 578 of Buenos Aires' 1,616 images have no confirmation method
recorded in ISIC's metadata.

**Hypothesis, not tested.** MSK biopsied nearly every lesion (97.7% histopathology, nevi
included). Benign lesions that were biopsied were presumably suspicious enough to biopsy, so
MSK's benign cases may be harder to tell from melanoma than benign cases confirmed by
follow-up elsewhere. That may explain MSK's low triage specificity (39.1% against 52.3%
pooled). Nothing here tests it. Evidence: `results/phase2/label_provenance/label_provenance.json`
(fetched 2026-10-05 from the ISIC public API; HAM10000's own `dx_type`).

## Finding: Mahalanobis scores also differ between GPU and CPU

The second batch-drift reference job (run 186825054715071, CPU, spec amendment 2) refit the
shipped Mahalanobis detector on CPU training features for v2-final-seed0 and compared its
scores with the drift run's GPU scores on 300 benchmark evaluation images (scores only; no
metric). The method-D PCA size reproduced exactly (541), but the scores did not meet the check
committed in advance: Spearman rank correlation 0.9988 (CIFAR-10) and 0.9988 (PathMNIST) against
a required 0.999; only 61% of images within 1% relative difference against a required 99%;
largest difference 5.0%. The job stopped as specified and the check was not loosened.
Batch-level drift detection was moved to Phase 3, where the baseline and the batches are both
scored on the same CPU setup.

The images are ranked almost identically, but individual scores move by up to about 5%, which
is enough to matter for a test that compares score distributions directly. Together with
Gram's 11.8% normalizer difference, this means a drift baseline and the batches compared with
it must be scored on the same hardware and settings. The cause is still an unverified
hypothesis (TF32 convolution arithmetic on the A10 GPU against CPU float32). Every Phase 2
detection metric compares scores from one GPU run, so none is affected. Evidence:
`results/phase2/batch_drift/reference-186825054715071/v2-final-seed0_check.json`.

## Simulated local recalibration at Buenos Aires (evidence for the change-control plan)

Spec committed before running (`results/phase2/ba_recalibration/spec.json`); not a change to the
shipped model or threshold; triage framing, not a clinical claim. Buenos Aires' 1,458 labelled
images (253 melanomas, 1,155 lesions) were re-scored with the shipped ensemble on CPU, which
reproduced the triage run's Buenos Aires numbers exactly. Over 20 random 50/50 lesion splits, a
threshold set on one half for 95% sensitivity, reported on the other half:

| Threshold | Sensitivity | Specificity | Referral rate |
| --- | --- | --- | --- |
| Local (set on half A; mean 0.0154) | 94.6% (range 88.9%–97.8%) | 23.7% (range 19.0%–32.0%) | 79.5% (range 71.6%–84.1%) |
| Shipped (0.0638, set on client validation) | 81.7% (range 77.3%–88.1%) | 59.1% (range 56.6%–63.1%) | 48.0% (range 44.8%–49.5%) |

Lowering the threshold at the new site brings sensitivity back near the target (below 95% in
10 of 20 splits), but specificity falls from about 59.1% to about 23.7%: the model ranks
Buenos Aires melanomas poorly, so recovering them by threshold alone means referring about four
in five images. For the change-control plan, site-level recalibration is therefore not enough on
its own at a site like this; retraining with the site's data is the modification that could
address it. Evidence: `results/phase2/ba_recalibration/results.json`.
