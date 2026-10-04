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
