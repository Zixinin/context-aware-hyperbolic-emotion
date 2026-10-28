# EXP5A_CLS_MEAN_MULTISEED_SUMMARY

Three paired seeds: 42 (frozen reuse), 123 and 2026 (new independent raw-BERT training). Fixed threshold 0.30.

## Test metrics: mean ± sample SD (%)

| Metric | CLS | Mean pooling | Paired Mean - CLS (pp) | Positive / negative / zero seeds |
|---|---:|---:|---:|---:|
| Micro-F1 | 59.7348 ± 1.1344 | 58.8948 ± 1.4171 | -0.8400 ± 1.0921 | 0 / 3 / 0 |
| Macro-F1 | 51.4988 ± 0.8412 | 51.0678 ± 0.7630 | -0.4311 ± 1.2324 | 1 / 2 / 0 |
| Macro Precision | 52.3575 ± 1.2467 | 51.7507 ± 2.1516 | -0.6068 ± 2.5462 | 1 / 2 / 0 |
| Macro Recall | 53.0491 ± 0.2060 | 52.6085 ± 0.7519 | -0.4406 ± 0.9468 | 1 / 2 / 0 |
| Weighted-F1 | 59.3811 ± 0.7903 | 58.5938 ± 0.9756 | -0.7873 ± 1.0148 | 0 / 3 / 0 |
| Rare-F1 | 44.7661 ± 1.9799 | 45.1047 ± 3.0668 | 0.3386 ± 4.4938 | 2 / 1 / 0 |

## Per-seed formal test results (%)

| Seed | Model | Micro-F1 | Macro-F1 | Rare-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| 42 | cls | 58.5859 | 50.9440 | 44.5968 | 58.5028 |
| 42 | mean | 58.5508 | 51.8691 | 48.1653 | 58.4954 |
| 123 | cls | 59.7645 | 51.0857 | 42.8762 | 59.6058 |
| 123 | mean | 57.6814 | 50.3499 | 45.1170 | 57.6711 |
| 2026 | cls | 60.8540 | 52.4668 | 46.8252 | 60.0347 |
| 2026 | mean | 60.4523 | 50.9843 | 42.0317 | 59.6148 |

## Paired deltas: Mean - CLS (pp)

| Seed | Micro-F1 | Macro-F1 | Rare-F1 |
|---|---:|---:|---:|
| 42 | -0.0350 | +0.9251 | +3.5685 |
| 123 | -2.0831 | -0.7358 | +2.2408 |
| 2026 | -0.4018 | -1.4825 | -4.7935 |

## Validation-selected checkpoint diagnostics

| Seed | Model | Best epoch | Validation Macro-F1 (%) | Validation Micro-F1 (%) | Trainable params |
|---|---|---:|---:|---:|---:|
| 42 | cls | 4 | 52.5711 | 58.7895 | 107741212 |
| 42 | mean | 4 | 51.8137 | 58.5872 | 107741212 |
| 123 | cls | 3 | 51.6797 | 59.8306 | 107741212 |
| 123 | mean | 4 | 50.9569 | 57.7874 | 107741212 |
| 2026 | cls | 3 | 51.8126 | 59.3745 | 107741212 |
| 2026 | mean | 3 | 51.6261 | 59.7770 | 107741212 |

## All 28 labels: mean F1 (%) and paired differences

| Label | Support | CLS mean±SD | Mean mean±SD | Mean - CLS (pp) | Mean better seeds |
|---|---:|---:|---:|---:|---:|
| admiration | 504 | 68.3527 ± 0.4793 | 68.0843 ± 0.1407 | -0.2684 | 1/3 |
| amusement | 264 | 83.1013 ± 0.4538 | 81.3686 ± 0.3573 | -1.7327 | 0/3 |
| anger | 198 | 51.4998 ± 0.2918 | 49.6857 ± 1.2522 | -1.8141 | 0/3 |
| annoyance | 320 | 37.9312 ± 1.1342 | 36.1628 ± 2.1716 | -1.7683 | 0/3 |
| approval | 351 | 39.8074 ± 0.1546 | 39.8272 ± 0.1573 | +0.0197 | 1/3 |
| caring | 135 | 42.5988 ± 1.0008 | 43.7904 ± 1.2341 | +1.1916 | 2/3 |
| confusion | 153 | 43.4924 ± 2.4922 | 43.6995 ± 2.1511 | +0.2071 | 1/3 |
| curiosity | 284 | 54.0450 ± 2.2019 | 54.6594 ± 1.2890 | +0.6144 | 2/3 |
| desire | 83 | 48.6467 ± 1.2336 | 45.8663 ± 1.6707 | -2.7803 | 0/3 |
| disappointment | 151 | 30.4577 ± 2.3067 | 30.7226 ± 1.5521 | +0.2649 | 2/3 |
| disapproval | 267 | 41.3108 ± 0.7975 | 40.2025 ± 1.8633 | -1.1083 | 1/3 |
| disgust | 123 | 49.7318 ± 1.1518 | 48.9521 ± 2.8612 | -0.7798 | 1/3 |
| embarrassment (rare) | 37 | 48.2556 ± 3.6403 | 47.1230 ± 0.4296 | -1.1326 | 2/3 |
| excitement | 103 | 44.3617 ± 4.3332 | 41.6418 ± 1.9084 | -2.7199 | 1/3 |
| fear (rare) | 78 | 68.5872 ± 0.3318 | 67.0478 ± 4.6405 | -1.5393 | 1/3 |
| gratitude | 352 | 91.3855 ± 1.0002 | 90.4893 ± 0.4095 | -0.8962 | 0/3 |
| grief (rare) | 6 | 13.3333 ± 23.0940 | 8.3333 ± 14.4338 | -5.0000 | 0/3 |
| joy | 161 | 61.5345 ± 0.5471 | 60.9569 ± 1.8779 | -0.5777 | 1/3 |
| love | 238 | 81.1257 ± 0.9233 | 79.0747 ± 0.2377 | -2.0510 | 0/3 |
| nervousness (rare) | 23 | 39.5662 ± 7.1329 | 35.0073 ± 3.5917 | -4.5589 | 1/3 |
| optimism | 186 | 57.5710 ± 0.4927 | 56.0186 ± 2.9947 | -1.5524 | 1/3 |
| pride (rare) | 16 | 47.9469 ± 5.5464 | 50.0000 ± 3.8462 | +2.0531 | 2/3 |
| realization | 145 | 24.6689 ± 2.2584 | 24.4478 ± 1.2262 | -0.2211 | 1/3 |
| relief (rare) | 11 | 28.6871 ± 13.3607 | 41.0831 ± 7.0326 | +12.3960 | 2/3 |
| remorse (rare) | 56 | 66.9864 ± 0.5538 | 67.1383 ± 2.6575 | +0.1519 | 1/3 |
| sadness | 156 | 54.6918 ± 2.5896 | 57.8751 ± 1.9341 | +3.1833 | 3/3 |
| surprise | 141 | 55.0861 ± 4.0609 | 54.5922 ± 1.6365 | -0.4939 | 1/3 |
| neutral | 1787 | 67.2043 ± 1.3570 | 66.0476 ± 1.7929 | -1.1566 | 1/3 |

## Grief + relief contributions (pp)

| Seed | Macro delta | Grief+relief part | Other26 part | Rare delta | Grief+relief part | Other5 rare part |
|---|---:|---:|---:|---:|---:|---:|
| 42 | +0.9251 | +0.4454 | +0.4796 | +3.5685 | +1.7818 | +1.7867 |
| 123 | -0.7358 | +0.6603 | -1.3961 | +2.2408 | +2.6411 | -0.4003 |
| 2026 | -1.4825 | -0.3133 | -1.1692 | -4.7935 | -1.2531 | -3.5404 |
| mean across seeds | -0.4311 | +0.2641 | -0.6952 | +0.3386 | +1.0566 | -0.7180 |

Signed contributions are primary; fraction ratios can exceed100% or change sign when effects offset. Grief test support=6, relief=11; do not equate gains on these labels with general long-tail improvement.

## Interpretation and scientific boundary

- Macro-F1: paired average delta -0.4311 pp, sample SD 1.2324 pp; positive in 1/3, negative in 2/3.
- Rare-F1: paired average delta +0.3386 pp, sample SD 4.4938 pp; positive in 2/3, negative in 1/3.
- Micro-F1: paired average delta -0.8400 pp, sample SD 1.0921 pp; positive in 0/3, negative in 3/3.

Signs are mixed or tied across seeds/metrics; the seed42 direction should not be presented as a stable architecture advantage.

- All models use fixed data/protocol/head/dropout; only representation and random seed differ. Each model independently starts from raw pretrained BERT, never from seed42/fine-tuned weights.
- Every best checkpoint uses validation Macro-F1 at0.30; earliest ties. Test is evaluated once per new full model. Secondary thresholds are validation-only diagnostics.
- Report averages per-seed metrics (sample SD ddof=1), not metrics of ensembled/pooled predictions. Paired-delta SD is computed directly across paired differences.
- All seeds contribute regardless of which wins. No test-driven parameter/checkpoint/seed selection. Validation and test ranks may disagree; inspect the validation diagnostics rather than declaring a final model from test scores.
- Only CLS and Mean receive new seeds. Query remains a seed42 pilot; no Query multi-seed mean or stable Query conclusion is claimed.
- Three seeds provide limited repeatability evidence, not proof of equivalence, significance or mechanism. No automatic new method/seed or Exp5B.
