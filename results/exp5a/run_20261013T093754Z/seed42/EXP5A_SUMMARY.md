# EXP5A_SUMMARY

## Research question

Does replacing a single global `[CLS]` representation with label-specific representations obtained through learnable label queries improve fine-grained multi-label emotion recognition on GoEmotions under an otherwise matched BERT training setup?

For this seed42 pilot, label-specific representations improved fixed-threshold Macro-F1 relative to the verified Exp1 reference.

> Scope: this is a single-seed Euclidean label-aware baseline. It is not a final model or a multi-seed conclusion.

## Fixed-threshold 0.30 comparison

| Metric | Exp1 Standard BERT | Exp5A Label-aware | Delta |
|---|---:|---:|---:|
| Micro-F1 | 0.587175 | 0.584308 | -0.002867 |
| Macro-F1 | 0.486221 | 0.514940 | +0.028719 |
| Macro Precision | 0.486890 | 0.524975 | +0.038085 |
| Macro Recall | 0.507001 | 0.529827 | +0.022826 |
| Weighted-F1 | N/A in verified Exp1 artifact | 0.584180 | N/A |
| Rare-F1 | 0.360010 | 0.464252 | +0.104241 |

The validation-selected global threshold is retained only as a secondary diagnostic; it is not the formal cross-experiment comparison.

## Largest per-label changes

### Top 10 improvements

| Label | Support | Exp1 F1 | Exp5A F1 | Delta |
|---|---:|---:|---:|---:|
| relief | 11 | 0.000000 | 0.400000 | +0.400000 |
| grief | 6 | 0.000000 | 0.250000 | +0.250000 |
| pride | 16 | 0.380952 | 0.480000 | +0.099048 |
| nervousness | 23 | 0.285714 | 0.347826 | +0.062112 |
| disappointment | 151 | 0.291971 | 0.347518 | +0.055547 |
| desire | 83 | 0.480000 | 0.521127 | +0.041127 |
| joy | 161 | 0.578035 | 0.611276 | +0.033241 |
| disgust | 123 | 0.489451 | 0.510823 | +0.021371 |
| sadness | 156 | 0.542169 | 0.561056 | +0.018887 |
| annoyance | 320 | 0.360176 | 0.375740 | +0.015564 |

### Top 10 regressions

| Label | Support | Exp1 F1 | Exp5A F1 | Delta |
|---|---:|---:|---:|---:|
| embarrassment | 37 | 0.508475 | 0.416667 | -0.091808 |
| excitement | 103 | 0.448980 | 0.411215 | -0.037765 |
| surprise | 141 | 0.563177 | 0.538462 | -0.024715 |
| amusement | 264 | 0.830189 | 0.811847 | -0.018342 |
| confusion | 153 | 0.445682 | 0.433604 | -0.012078 |
| neutral | 1787 | 0.666124 | 0.656004 | -0.010119 |
| anger | 198 | 0.497738 | 0.489510 | -0.008227 |
| admiration | 504 | 0.681239 | 0.673250 | -0.007989 |
| remorse | 56 | 0.685714 | 0.680851 | -0.004863 |
| approval | 351 | 0.401606 | 0.396846 | -0.004760 |

## Protocol and limits

- Best checkpoint: epoch 4, selected only by validation Macro-F1 at fixed 0.30.
- Test set was evaluated once after checkpoint and threshold selection were frozen.
- Same official splits, 28-label order, max length 50, four epochs, batch 16, AdamW 5e-5, weight decay 0.01, linear warmup 0.1 and seed42 as Exp1.
- The only intended model change is token-level label-wise attention with 28 learned queries and per-label linear heads.
- No hierarchy, LGSA, graph, hyperbolic geometry, class weighting, resampling, contrastive/prototype loss or test-set tuning.
- Exp1 comparison reads three present artifacts whose SHA-256 values were verified against the reference manifest: rare_label_metrics.json, test_metrics.json, test_per_label_fixed_0.30.csv.
- Historical Exp1 manifest note: 8 copied files match archived SUCCESS.json SHA256; live Slurm accounting was not re-queried in this Ubuntu session.
- Suggested next step only if healthy: Exp5B Euclidean label prototypes. Do not execute it from this run.
