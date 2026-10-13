# Experiment 5A - Label-aware Euclidean Baseline

## Research question

Does replacing one global BERT `[CLS]` representation with 28 label-specific
representations, learned by label-query attention over token states, improve
GoEmotions multi-label classification under an otherwise matched Exp1 protocol?

This is a clean seed42 pilot, not a final model and not a multi-seed conclusion.

## Fixed scientific boundary

- Standard raw `bert-base-cased`, revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`.
- Official simplified GoEmotions splits: 43,410 / 5,426 / 5,427 and the canonical 28-label order.
- Four epochs, max length 50, effective batch 16, AdamW LR `5e-5`, weight decay `0.01`, linear warmup `0.1`, FP32, seed42.
- Checkpoint selection uses validation Macro-F1 at fixed threshold 0.30; exact ties keep the earlier epoch.
- Fixed threshold 0.30 is the formal test comparison. A validation-selected global threshold is secondary only.
- BCEWithLogitsLoss is the only loss.
- The only intended model change is 28 learned queries, scaled dot-product label-wise attention, and 28 independent per-label linear heads.
- No LGSA, hierarchy, graph/GNN/GAT, hyperbolic geometry, prototypes, contrastive loss, class weights, resampling, augmentation, extra transformer or cross-label self-attention.

The code reads the already-verified generic assets and Exp1 reference artifacts,
but it does not import any LGSA or hierarchy implementation.

## Ubuntu execution order

All commands must use `.venv-bert/bin/python` through `launch.sh`. Do not use the
currently activated shell `python`, which may point to another project.

```bash
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
bash -n experiments/exp5_label_aware_euclidean/launch.sh
bash experiments/exp5_label_aware_euclidean/launch.sh audit
```

After the audit marker, test ntfy and confirm the phone received it:

```bash
bash experiments/exp5_label_aware_euclidean/launch.sh notify-test
```

Then run the engineering smoke in `tmux`, using the exact run directory printed
by the audit:

```bash
bash experiments/exp5_label_aware_euclidean/launch.sh smoke /absolute/results/exp5a/run_TIMESTAMP
```

Only after the smoke artifacts and completion marker are verified may the same
run directory be passed to the formal seed42 phase:

```bash
bash experiments/exp5_label_aware_euclidean/launch.sh full /absolute/results/exp5a/run_TIMESTAMP
```

Do not start seeds 123 or 2026. Do not tune hyperparameters in response to test
metrics. A failed run remains in its timestamped directory for diagnosis and is
never overwritten.

## Evidence levels

- `EXP5A_UNIT_TESTS_PASS`: synthetic engineering checks only.
- `EXP5A_PREFLIGHT_PASS`: environment/data/raw-weight/reference provenance checks.
- `EXP5A_SMOKE_COMPLETE_VERIFIED`: 1,000-train/500-validation engineering run; not a scientific result.
- `EXPERIMENT_5A_COMPLETE_VERIFIED`: seed42 full train/validation/test artifacts verified locally on D3020-A3.

The final summary explicitly records that the historical Exp1 archive artifacts
are hash-verified but their old Slurm job accounting was not re-queried during
this Ubuntu run.
