# Exp6C-1 verification

PASS: frozen A0 code/checkpoint/results; identical canonical IDs/27 labels/snapshot/tokenizer; historical A0/fresh shared initial tensors and initial train/eval logits; Formula A aux value/gradient equivalence; ordinary unweighted BCE; ball/finite/nonzero gradients; optimizer/scheduler and four epochs; validation-only earliest-tie selection; entire validation reload; single formal Test; saved-prediction aggregate/per-label/Rare-F1 recomputation and paired A0 comparison.

No extra seeds, distance weighting, new hierarchy, ASL, Graph or test tuning.
