# EXP6C0_SUMMARY

RoBERTa-base + 27-label multi-label BCE, seed42. Fixed threshold 0.30.

## 1. Data and retained label cardinality

See DATASET_AUDIT.md for per-label support and the exact canonical IDs in DATASET_MANIFEST.json.

- train: 30587 retained; 24820 single-emotion; 5767 multi-emotion; 12823 neutral-only excluded.
- validation: 3834 retained; 3122 single-emotion; 712 multi-emotion; 1592 neutral-only excluded.
- test: 3821 retained; 3154 single-emotion; 667 multi-emotion; 1606 neutral-only excluded.

## 2. Training completion

Completed 4 epochs and 7648 optimizer steps on NVIDIA GeForce RTX 4090 D.

## 3. Checkpoint

Epoch 3: validation Macro-F1 0.58135583; full validation reload verified before the only formal test.

## 4. Formal test metrics

| Metric | Value |
|---|---:|
| micro_precision | 0.60271814 |
| micro_recall | 0.67371202 |
| micro_f1 | 0.63624077 |
| macro_precision | 0.57724539 |
| macro_recall | 0.60223335 |
| macro_f1 | 0.57927600 |
| weighted_f1 | 0.63164630 |
| rare_f1 | 0.53316022 |

## 5. Best and worst emotion labels

Descriptive test results only; no tuning follows.

| Group | Emotion | Support | F1 |
|---|---|---:|---:|
| Best five | gratitude | 352 | 0.92351275 |
| Best five | amusement | 264 | 0.86703097 |
| Best five | love | 238 | 0.81835564 |
| Best five | admiration | 504 | 0.72792150 |
| Best five | curiosity | 284 | 0.71619614 |
| Worst five | excitement | 103 | 0.45643154 |
| Worst five | nervousness | 23 | 0.45000000 |
| Worst five | disappointment | 151 | 0.37062937 |
| Worst five | realization | 145 | 0.31924883 |
| Worst five | relief | 11 | 0.31578947 |

## 6. Fixed seven rare emotions

| Emotion | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| grief | 6 | 0.75000000 | 0.50000000 | 0.60000000 |
| pride | 16 | 0.66666667 | 0.37500000 | 0.48000000 |
| relief | 11 | 0.37500000 | 0.27272727 | 0.31578947 |
| nervousness | 23 | 0.52941176 | 0.39130435 | 0.45000000 |
| embarrassment | 37 | 0.48648649 | 0.48648649 | 0.48648649 |
| remorse | 56 | 0.56521739 | 0.92857143 | 0.70270270 |
| fear | 78 | 0.62886598 | 0.78205128 | 0.69714286 |

## 7. Differences from historical experiments

Earlier Exp5A/Exp6B used BERT-base-cased, all 28 outputs and the complete official splits including neutral-only. A0 uses RoBERTa-base, 27 emotion outputs and official split filtering. Exp6B also added Formula A auxiliary loss; A0 uses unweighted mean BCE only. The four-epoch, batch16, seed42, AdamW/ceil-warmup/clipping and fixed0.30 validation-selection budget is retained. No 27-vs-28 Macro-F1 subtraction is a controlled improvement estimate. Original single-label HypEmo Weighted-F1 66.3% is not comparable.

## 8. Suitability as the later matched baseline

All verification gates passed. This is a reusable single-seed baseline, not a multi-seed statistical conclusion. Preserve package hashes, model snapshot, dataset IDs, logits, classifier path and exact training/evaluation protocol for A1/A2.

## 9. Loss, predictions and gradients

All logged batch losses and clipping norms are finite; nonzero finite encoder/head gradients were checked in smoke and each epoch's first batch. Every step used error_if_nonfinite gradient clipping. These checks establish numerical validity, not the absence of all modeling limitations.
Total predicted positives: 5077; gold positives: 4542; mean predictions/example: 1.328710; zero-prediction examples: 45; zero-prediction labels: none. Per-label TP/FP/FN and prediction counts are archived.

## 10. Next-stage classification-path contract

A1/A2 must reuse model27.Roberta27.encoder and its complete first-token/dropout/classifier forward unchanged, the original initialization RNG convention, same snapshot/tokenizer, 27-label mapping/retained IDs, seed, loader, optimizer and validation rule. Any auxiliary component must be a separately controlled addition; do not replace the head with RobertaForSequenceClassification. The existing FYP independent-per-label dropout layout is documented in IMPLEMENTATION_AUDIT.md. Freeze baseline path initialization before any future auxiliary-module RNG draws. No A1/A2 code is executed in this experiment.

Supplementary neutral-only diagnostic: skipped (optional). Threshold search: not performed. Stop after Exp6C-0.
