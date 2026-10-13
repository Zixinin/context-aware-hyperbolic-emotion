# Exp3B diagnostics (analysis only)

This directory is independent of the frozen `experiments/exp3_dpt_peer` package.
There is no new model, optimizer, training, threshold search, lambda tuning or test inference.
The original dirty Git worktree, assets, output files, checkpoints and earlier diagnostics are preserved.

## Procedure

1. Human workstation gate: hostname / nvidia-smi / git status / git HEAD (completed).
2. Deploy only this new directory. Never overwrite an existing diagnostic directory.
3. Run `run.py --repo /home/zixinfeng/Documents/context-aware-hyperbolic-emotion --audit-only` with `.venv-lgsa/bin/python`.
4. After audit approval, run the same script without `--audit-only` in tmux. The script includes start/success/failure ntfy notifications for `zixin-fyp-4090-1119`. Notification failure is nonfatal and has a 5-second timeout.
5. Validate new `SUCCESS.json` and all recorded result hashes, then read the report. Notifications alone do not establish scientific completion.
6. Stop. No new experiment is launched or implemented.

The exact new output directory is `outputs/exp3b_diagnostics_seed42`. It must not exist before either audit or full diagnosis. Failure leaves partial new artifacts intact for investigation, never overwrites them.

## Evidence reused

- Exp2: `experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42`
- Exp3: `outputs/exp3_dpt_peer/seed42_20261011`
- Existing diagnostic: `outputs/exp3_diagnostics/seed42_20261011_v1`
- Validation replay: `outputs/exp3_diagnostics/validation_epoch3_4_20261011_v1`
- Frozen LGSA source, hierarchy, prepared dataset and pinned raw BERT assets.

Every existing result/checkpoint file in these directories is hashed before and after diagnosis. Existing success manifests are verified before use. Frozen LGSA and Exp3 source proofs must match the completed run. All asset files and both checkpoints are protected; checkpoint loading is CPU-only for provenance checks and the best model is then used on the GPU.

## Actual peer contributions

Historical steps saved scalar batch losses and pair counts, NOT individual term losses. Historical term-specific losses cannot be recovered exactly.

The independent snapshot therefore uses all 43,410 existing TRAIN rows once, `RandomState(42)` permutation, batch16, fixed best epoch3 weights, and per-batch RNG seed `420000 + batch_index`. `model.train()` is needed to activate the unchanged original supervised/dropout/MLM graph; `torch.no_grad()` prevents gradients for this forward-only pass. No optimizer exists.

Each selected term uses the actual original `PeerHardNegativeContrastiveLoss.select` and the same DPT-derived `positive_label_contrastive` function. Recomputed term loss means must match the native peer loss for each batch. All co-GT labels are excluded; neutral singleton skips. Zero-term batches remain in the per-batch mean denominator.

Raw contribution is `sum(term_loss)`. Weighted contribution is `sum(0.5 * term_loss / eligible_terms_in_that_batch)`. The weighted share denominator is all peer contributions, not original+peer total loss. Counts, loss magnitudes and gradient norms are different measurements. Snapshot loss tables are not historical optimization pressure.

## Gradient conflict

The first 10 batches of the predeclared TRAIN permutation are probed at best epoch3. Each batch's original and peer objectives use the same forward graph and stochastic draws. Original LGSA means MLM+classification+original GOF/HGA (alpha=1 at best step); peer is isolated, unweighted. Both are computed using `torch.autograd.grad`, never `.backward()` or `.step()`. Parameter `.grad` must remain empty.

Only trainable parameters connected to BOTH losses are included in the flattened shared gradient comparison. Frozen fine HGS/raw prompt rows, original-only decoder biases and disconnected parameters are excluded and listed. The shared parameter set must remain identical across all 10 batches. GAT and MLM anchor-transform scopes and optional all 12 BERT layers are included. FP64 streaming dot/norm accumulations avoid giant concatenated vectors. Positive lambda leaves cosine unchanged; weighted peer norms are separately reported before clipping.

Sample SD uses ddof=1. `cos<0`, `cos>0` and `abs(cos)<=0.05` are intentionally overlapping descriptive categories; strict disjoint bins below -0.05 / within +/-0.05 / above 0.05 are also saved. Undefined zero-norm cosines are unavailable, not zero.

## Classification and sibling metrics

All decisions remain at frozen 0.30. TP/FP/FN reconcile with saved metrics. Macro uses all 28 fine labels and zero_division=0; supported-only macro is also shown for subsets. P/R/F1 values and shares in CSV/JSON are fractions; report/figure pp values multiply by100. Sigmoid for descriptive matched metrics uses float64, as in the completed Exp3 comparison report. Missing score averages / correlations are null, not zero.

Structural label eligibility means parent has at least2 children; actual example eligibility means at least one GT has a NON-GT sibling after excluding all co-GT. These two populations have separate tables. Sibling metrics use the highest-logit eligible negative independently for each model, with ascending fine ID tie-break, same as the original reporting function. Validation cosine margins use the SAME logit-selected pair, not cosine-top-k.

Branch support is summed GT label instances, NOT unique examples (multi-label examples can contribute to several branches). All 28 labels are analysed; correlations use average tied ranks, no p-values/significance claims. Train frequency is obtained from TRAIN only. Test results are diagnostic, not authority to choose new pairs, thresholds, checkpoints, lambdas or losses.

## Decision limits

The report separates descriptive local insufficiency (C), snapshot gradient-conflict evidence (A) and concentration evidence (B). None establishes a unique causal mechanism. A strategically recommended STOP_DPT_BRANCH must be distinguished from a universal claim that DPT is ineffective. No follow-up code is created. The main FYP remains hierarchy / hyperbolic emotion representation.

`test_diagnostics.py` uses synthetic CPU data only. Passing local tests does not mean Ubuntu diagnosis ran. Runtime result manifests record input provenance, zero updates, exact model-state preservation and protected original-file hashes.
