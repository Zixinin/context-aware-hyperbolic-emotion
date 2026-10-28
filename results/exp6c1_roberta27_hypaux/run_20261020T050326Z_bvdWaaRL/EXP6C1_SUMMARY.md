# EXP6C1_SUMMARY

A1: RoBERTa27 + ordinary BCE + 0.1 x validated Formula A auxiliary-only objective.

The sole planned method addition is the original HNN/tangent decoder/fixed 27 label vectors and its multi-positive auxiliary supervision.
A0 code, checkpoint and results are read-only; no A0 retraining or test inference was performed.

## Matched protocol

Same canonical 30587/3834/3821 IDs, 27-label order, raw RoBERTa snapshot/tokenizer, exact A0 forward path, seed42, max_length50, batch16, four epochs/7648 steps/765 ceil warmup, AdamW5e-5/WD0.01, clipping1, FP32/eager, fixed0.30 and validation Macro-F1/earliest tie.
Shared initial tensors and initial train/eval classification logits match A0 exactly; see INITIALIZATION_PARITY.json. No A0 trained checkpoint or A1 smoke weights initialize formal training.

Best A1 epoch: 3; validation Macro-F1=0.56020868. Full 3834-example reload passed before the only formal test.

## Fixed0.30 Test comparison

| Metric | A0 | A1 | Delta (percentage points) |
|---|---:|---:|---:|
| micro_precision | 60.2718% | 62.4895% | +2.2177 |
| micro_recall | 67.3712% | 65.4337% | -1.9375 |
| micro_f1 | 63.6241% | 63.9277% | +0.3036 |
| macro_precision | 57.7245% | 58.5312% | +0.8067 |
| macro_recall | 60.2233% | 56.9279% | -3.2955 |
| macro_f1 | 57.9276% | 56.6248% | -1.3028 |
| weighted_f1 | 63.1646% | 63.4545% | +0.2899 |
| rare_f1 | 53.3160% | 46.1239% | -7.1921 |

At this single seed, A1 decreased matched test Macro-F1 by -1.3028 percentage points. This does not establish a multi-seed or geometry-specific causal conclusion.

## All 27 emotion labels

| Emotion | Support | A0 F1 | A1 Precision | A1 Recall | A1 F1 | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| admiration | 504 | 72.79% | 68.84% | 79.76% | 73.90% | +1.10 |
| amusement | 264 | 86.70% | 83.80% | 90.15% | 86.86% | +0.16 |
| anger | 198 | 59.17% | 56.19% | 64.14% | 59.91% | +0.73 |
| annoyance | 320 | 49.37% | 43.53% | 54.69% | 48.48% | -0.89 |
| approval | 351 | 56.06% | 54.42% | 54.42% | 54.42% | -1.65 |
| caring | 135 | 49.43% | 48.37% | 54.81% | 51.39% | +1.96 |
| confusion | 153 | 55.03% | 58.40% | 47.71% | 52.52% | -2.52 |
| curiosity | 284 | 71.62% | 64.35% | 78.17% | 70.59% | -1.03 |
| desire | 83 | 57.52% | 72.13% | 53.01% | 61.11% | +3.59 |
| disappointment | 151 | 37.06% | 40.60% | 35.76% | 38.03% | +0.97 |
| disapproval | 267 | 54.70% | 50.69% | 55.06% | 52.78% | -1.92 |
| disgust | 123 | 50.22% | 55.00% | 53.66% | 54.32% | +4.10 |
| embarrassment | 37 | 48.65% | 68.18% | 40.54% | 50.85% | +2.20 |
| excitement | 103 | 45.64% | 52.75% | 46.60% | 49.48% | +3.84 |
| fear | 78 | 69.71% | 57.41% | 79.49% | 66.67% | -3.05 |
| gratitude | 352 | 92.35% | 93.90% | 91.76% | 92.82% | +0.46 |
| grief | 6 | 60.00% | 0.00% | 0.00% | 0.00% | -60.00 |
| joy | 161 | 65.67% | 63.64% | 60.87% | 62.22% | -3.45 |
| love | 238 | 81.84% | 78.87% | 87.82% | 83.10% | +1.27 |
| nervousness | 23 | 45.00% | 55.56% | 43.48% | 48.78% | +3.78 |
| optimism | 186 | 58.64% | 61.76% | 56.45% | 58.99% | +0.35 |
| pride | 16 | 48.00% | 87.50% | 43.75% | 58.33% | +10.33 |
| realization | 145 | 31.92% | 49.37% | 26.90% | 34.82% | +2.90 |
| relief | 11 | 31.58% | 40.00% | 18.18% | 25.00% | -6.58 |
| remorse | 56 | 70.27% | 60.47% | 92.86% | 73.24% | +2.97 |
| sadness | 156 | 55.58% | 54.50% | 66.03% | 59.71% | +4.13 |
| surprise | 141 | 59.49% | 60.14% | 60.99% | 60.56% | +1.07 |

Rare-F1 uses the same seven named emotions as A0. Detailed rare TP/FP/FN/precision/recall/F1 appear in rare_label_metrics.json.

## Auxiliary scale and gradient diagnostics

See LOSS_SPECIFICATION.md for the exact q-positive mean/batch mean formula and paths. No distance weights, positive class weights, sample weights or alternative formula are used. LOSS_AND_GRADIENT_CASES.json in smoke verifies single/multiple/neutral-coincident retained examples, Formula A value and gradient equivalence, unweighted BCE, fixed label vectors, sphere-interior values, and BCE/auxiliary-to-shared-encoder gradients.

Training epoch means and each batch's BCE, raw/weighted auxiliary, signed/absolute ratios, CE/dot decomposition, ball norm and clipping norm are archived in training_log.csv and batch_log.csv. First-batch gradients for all four module groups are saved each epoch.

| Evaluation split | BCE | Raw auxiliary | Weighted auxiliary | Total |
|---|---:|---:|---:|---:|
| validation | 0.08561485 | 1.52728934 | 0.15272893 | 0.23834379 |
| test | 0.08456518 | 1.52376396 | 0.15237640 | 0.23694158 |

All numerical checks passed; this is not a claim that the auxiliary dominates or benefits every class. Raw auxiliary values may be negative due to the dot term; they are not clamped, normalized or tuned. Original HNN dropout consumes additional RNG during training, so subsequent shared dropout masks need not match A0. This compares addition of the complete auxiliary branch/supervision; isolating hyperbolic geometry itself would require a separately authorized control.

## Completion boundary

No extra seed, distance weighting, ASL, graph, new hierarchy or test tuning. Stop after Exp6C-1. All four epochs, validation-only selection, complete validation reload, one test and saved-prediction verification passed.
