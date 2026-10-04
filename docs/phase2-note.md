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
referral rates above describe these datasets only and would be much lower, and mostly made of
false referrals, in primary care. No comparison is made with any marketed device. As context
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
