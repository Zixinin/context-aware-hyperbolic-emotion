# EXP5A_CONTROL_SUMMARY

Seed42 only; fixed threshold 0.30. No test-driven checkpoint selection.

## Formal comparison

| Metric | Historical Exp1 (reference only) | Same-env raw CLS | Masked mean | Frozen label query |
|---|---:|---:|---:|---:|
| Micro-F1 | 0.587175 | 0.585859 | 0.585508 | 0.584308 |
| Macro-F1 | 0.486221 | 0.509440 | 0.518691 | 0.514940 |
| Macro Precision | 0.486890 | 0.513112 | 0.535552 | 0.524975 |
| Macro Recall | 0.507001 | 0.528220 | 0.532874 | 0.529827 |
| Weighted-F1 | N/A in verified artifact | 0.585028 | 0.584954 | 0.584180 |
| Rare-F1 | 0.360010 | 0.445968 | 0.481653 | 0.464252 |

Micro precision/recall and weighted precision/recall are also saved in each test_metrics.json and CONTROL_COMPARISON.json.

## Paired deltas (percentage points)

| Metric | Mean - CLS | Query - Mean | Query - CLS |
|---|---:|---:|---:|
| Micro-F1 | -0.0350 | -0.1200 | -0.1550 |
| Macro-F1 | +0.9251 | -0.3751 | +0.5499 |
| Macro Precision | +2.2440 | -1.0576 | +1.1863 |
| Macro Recall | +0.4654 | -0.3047 | +0.1607 |
| Weighted-F1 | -0.0074 | -0.0774 | -0.0848 |
| Rare-F1 | +3.5685 | -1.7401 | +1.8283 |

## Checkpoint and parameter diagnostics

| Model | Best epoch | Validation Macro-F1 | Validation Micro-F1 | Trainable parameters |
|---|---:|---:|---:|---:|
| cls | 4 | 0.525711 | 0.587895 | 107741212 |
| mean | 4 | 0.518137 | 0.585872 | 107741212 |
| query | 4 | 0.521196 | 0.585854 | 107762716 |

## All 28 labels

| Label | Support | CLS F1 | Mean F1 | Query F1 | Mean - CLS | Query - Mean |
|---|---:|---:|---:|---:|---:|---:|
| admiration | 504 | 0.678182 | 0.680394 | 0.673250 | +0.002212 | -0.007144 |
| amusement | 264 | 0.825784 | 0.815199 | 0.811847 | -0.010585 | -0.003352 |
| anger | 198 | 0.512941 | 0.503529 | 0.489510 | -0.009412 | -0.014019 |
| annoyance | 320 | 0.366667 | 0.340741 | 0.375740 | -0.025926 | +0.034999 |
| approval | 351 | 0.398448 | 0.397892 | 0.396846 | -0.000556 | -0.001046 |
| caring | 135 | 0.427046 | 0.442029 | 0.411765 | +0.014983 | -0.030264 |
| confusion | 153 | 0.448753 | 0.413978 | 0.433604 | -0.034775 | +0.019626 |
| curiosity | 284 | 0.530735 | 0.547826 | 0.542986 | +0.017091 | -0.004840 |
| desire | 83 | 0.472222 | 0.472222 | 0.521127 | +0.000000 | +0.048905 |
| disappointment | 151 | 0.330986 | 0.325088 | 0.347518 | -0.005898 | +0.022429 |
| disapproval | 267 | 0.404110 | 0.395349 | 0.385621 | -0.008761 | -0.009728 |
| disgust | 123 | 0.503937 | 0.487179 | 0.510823 | -0.016758 | +0.023643 |
| embarrassment (rare) | 37 | 0.461538 | 0.468750 | 0.416667 | +0.007212 | -0.052083 |
| excitement | 103 | 0.403846 | 0.424528 | 0.411215 | +0.020682 | -0.013313 |
| fear (rare) | 78 | 0.682635 | 0.723926 | 0.674419 | +0.041292 | -0.049508 |
| gratitude | 352 | 0.912676 | 0.902954 | 0.906516 | -0.009722 | +0.003562 |
| grief (rare) | 6 | 0.400000 | 0.250000 | 0.250000 | -0.150000 | +0.000000 |
| joy | 161 | 0.616279 | 0.612245 | 0.611276 | -0.004034 | -0.000969 |
| love | 238 | 0.800774 | 0.788350 | 0.796905 | -0.012424 | +0.008556 |
| nervousness (rare) | 23 | 0.340426 | 0.391304 | 0.347826 | +0.050879 | -0.043478 |
| optimism | 186 | 0.572152 | 0.567775 | 0.548718 | -0.004377 | -0.019057 |
| pride (rare) | 16 | 0.416667 | 0.461538 | 0.480000 | +0.044872 | +0.018462 |
| realization | 145 | 0.225256 | 0.241135 | 0.237762 | +0.015879 | -0.003373 |
| relief (rare) | 11 | 0.153846 | 0.428571 | 0.400000 | +0.274725 | -0.028571 |
| remorse (rare) | 56 | 0.666667 | 0.647482 | 0.680851 | -0.019185 | +0.033369 |
| sadness | 156 | 0.543210 | 0.581250 | 0.561056 | +0.038040 | -0.020194 |
| surprise | 141 | 0.512111 | 0.555556 | 0.538462 | +0.043445 | -0.017094 |
| neutral | 1787 | 0.656435 | 0.656563 | 0.656004 | +0.000128 | -0.000559 |

## Grief + relief contribution (percentage points)

| Contrast | Macro delta | Grief+relief part | Other26 part | Rare delta | Grief+relief part | Other5 rare part |
|---|---:|---:|---:|---:|---:|---:|
| Mean - CLS | +0.9251 | +0.4454 | +0.4796 | +3.5685 | +1.7818 | +1.7867 |

Mean - CLS: grief+relief account for 48.15% of net Macro change; 13 labels improve and 14 regress.
| Query - Mean | -0.3751 | -0.1020 | -0.2731 | -1.7401 | -0.4082 | -1.3320 |

Query - Mean: grief+relief account for 27.20% of net Macro change; 9 labels improve and 18 regress.
| Query - CLS | +0.5499 | +0.3434 | +0.2065 | +1.8283 | +1.3736 | +0.4547 |

Query - CLS: grief+relief account for 62.44% of net Macro change; 13 labels improve and 15 regress.

Contributions are signed; ratios can be negative or exceed 100% when other labels offset them. Raw signed contributions are primary. Grief support=6 and relief support=11; concentrated gains are not evidence of general long-tail improvement.

## Interpretation and decision boundary

No arbitrary post-hoc 'clearly better' or 'approximately equal' tolerance is used. This seed42 diagnostic supports numerical attribution clues, not statistical consistency, equivalence or causal proof.

Observed Macro-F1: Mean - CLS = +0.9251 pp; Query - Mean = -0.3751 pp.

- If mean substantially beats CLS while query and mean remain practically close across metrics/labels, results are compatible with pooling explaining most gains. Do not claim statistical equivalence from one seed.
- If query exceeds mean across important metrics and many labels, this suggests incremental query-training value; multi-seed evidence is needed before calling it consistent.
- If all are close, no robust representation advantage is established; do not automatically start more seeds.

## Matched implementation and limits

- A uses raw last_hidden_state[:,0,:], not historical Exp1's learned tanh pooler; Exp1 is reference only.
- B averages every unmasked token, including CLS/SEP, matching uniform attention in C. PAD is excluded.
- A/B use the same classifier weights/bias and per-label independent dropout layout as frozen C; this is mathematically a D-to-28 linear head at evaluation. A shared [B,D] dropout would introduce an extra training difference, so is not used.
- One discarded initialization draw (not a model parameter/module) matches C's initial classifier and CPU/CUDA RNG states, verified on the target environment. CLS/mean have no queries or token-attention modules.
- Data loader, tokenization, shuffle seed, optimizer grouping, sample-normalized gradients, scheduler/warmup, validation metrics and threshold code are read-only reused from frozen C. Four epochs / seed42 only / BCE only.
- C is reused, never retrained. All C hashes are verified and formal metrics recomputed on CPU from saved logits and canonical test labels; no new GPU test pass for C.
- Best epoch is highest validation Macro-F1 at 0.30, earliest on ties. Each control evaluates test only once after checkpoint selection. Secondary threshold diagnostics are validation-selected and not the formal comparison.
- Independent training changes backbone optimization trajectories; this comparison does not isolate an inference-only attention intervention. Historical C's training sample_ids prove dataset membership/order, not a stored per-epoch permutation trace.
- Earlier inference diagnostic was 128 validation texts / 3584 label decisions, not 3584 examples. It does not prove queries have no training effect.
- No prototype, ASL, focal/class-weight/contrastive loss, graph, hierarchy, hyperbolic geometry, label semantics or extra transformer. No Exp5B or extra seeds are started.
