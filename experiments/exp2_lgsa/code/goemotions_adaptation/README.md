# Experiment 2: LGSA-GoEmotions

This is a separately version-controlled adapter around the tested minimal LGSA
parent-map patch. It does not modify `original_lgsa` or `working_lgsa`.

## Fixed first-run protocol

- Name: **GoEmotions sentiment-derived hierarchy v0**. Positive, Negative,
  Ambiguous grouping comes from the archived Google sentiment mapping; the
  Neutral → neutral singleton is an explicitly added engineering branch.
  Coarse labels are derived supervision, not original ground-truth annotations.
- Official splits: 43,410 / 5,426 / 5,427. Read the existing baseline dataset
  assets without altering them. Preserve IDs, text, case, and multi-label targets.
- Fine IDs 0–27; coarse IDs 28–31; logical ROOT is not a target; depth graph
  nodes 32–33; task MASK is outside the graph. A sample can have several parents.
- Only 28 fine logits at depth 1 are evaluated. Coarse classes never enter F1.
- Raw `google-bert/bert-base-cased`, revision
  `cd5ef92a9fb2f889e972770a36d4ed042daf221e`. Never load baseline/WoS/smoke-trained
  weights into a new full run. No automatic uncased fallback.
- Smoke: deterministic 1,000 train / 500 validation, seed 42, one epoch;
  test is not evaluated or used to choose thresholds/checkpoints.
- Full: seed 42, four epochs, batch 16, fp32, eager attention, zero loader workers.
  Same text budget as Experiment 1: 50 tokens including CLS/SEP, then replace
  terminal SEP by 2 depth prompts + 2 task MASKs + SEP, giving total length 54.
- Retain upstream GAT (one layer), MLM and multi-label categorical CE, and four
  HGA terms. No BCE replacement, extra GNN, loss ablation or hyperbolic component.
- Adam LR 1e-4, weight decay 0; 10% linear warmup then linear LR decay;
  clip grad norm at 1.0. HGA weight 1, angle 10 degrees, decay .9, linear HGA
  ramp over first 10% updates. These optimization choices are fixed before test.
  Baseline used AdamW 5e-5/weight decay .01. Upstream used Adam with dataset-specific
  long warmup/exponential schedule. This is a budget-matched adaptation, not a
  strict reproduction or controlled geometry-only ablation.
- Retain upstream multi-label CE's actual flattening of depth masks into one
  32-active-label row per sample. Do not silently substitute separate depth CE.
- Preserve upstream MLM masking, including its random replacement ID lower bound
  104. Keep frozen GraphEmbedding semantics. `align_w` is not used by the official
  active loop and is not introduced as a new loss here.
- Checkpoint selection: highest validation Macro-F1 at .30, tie keeps earlier
  epoch. Setting A: fixed .30. Setting B: validation-selected global threshold
  .05 through .90 at .05 increments, highest macro-F1, smallest threshold on ties.
  Only after selection evaluate test. Also record native logit > 0 diagnostics.
- Rare IDs come only from complete train counts: [16,21,23,19,12,24,14].

## Hard success gates

1. Slurm `afterok:2985243`; current patched WoS code already asserts mapping,
   changed fine vectors, orthogonality, rank and norm/projection preservation.
   Any exception exits nonzero; SUCCESS is written only after all checks.
2. Read-only upstream gate checks specific job/commit, matching finalized JSON,
   geometry, checkpoint hash, training steps, reload proof and source integrity.
3. GoEmotions independently checks pre-GAT HGS map, fine-vector changes, top and
   sibling residual absolute cosine <=1e-4, full rank, norm/projection/residual
   preservation rtol1e-5/atol1e-6. Neutral singleton is recorded separately.
4. Finite losses/gradient norm throughout; explicit BERT/GAT gradient audit,
   actual parameter update, full validation, save/reload logits agreement.
5. Standalone `accept_smoke.py` rejects missing/failed/tampered evidence or changes
   to code, configuration, backbone or dataset. It returns nonzero on failure.
6. Full starts in a fresh Python process. Its raw-backbone and complete initialized
   LGSA state fingerprints must equal the smoke's PRE-training fingerprints.
   No checkpoint argument is accepted for initialization.

Both processes use one GPU allocation with `set -euo pipefail`; full cannot run
if smoke or its acceptance fails. Failure of upstream prevents dependent launch.
The dependency does not reserve a GPU or promise an immediate start.

## Run and recovery

The outer `slurm/paste_submit_goemotions_chain.sh` is the self-contained command
for the user to paste on HPC. It validates its package hash, creates/commits this
repository on branch `goemotions-adaptation`, prepares assets, runs unit tests and
a tiny RANDOM CPU BERT/GAT integration check, then submits the GPU chain.
No real pretrained training occurs on login nodes or the Mac.

12-hour allocation, one RTX4090, four CPUs, 20,000 MB memory, email BEGIN/END/FAIL
to Zixin.Feng23@student.xjtlu.edu.cn. A new job ID is printed; local validation is
not evidence of a successful GPU smoke/full run.

If Slurm has purged the completed upstream dependency record, submission may
proceed only after `sacct` main AND .batch show COMPLETED/0:0 and all upstream
artifact/source gates pass; this is recorded distinctly. Failed/unknown upstream
states never fall back to unconditional submission. No `afterany` is used.

Full checkpoints are atomically written every 200 steps and at epoch boundaries.
An interruption loses at most work since the last durable checkpoint. Resume
requires the same run directory, model, code, dataset, config and phase=full;
optimizer, scheduler and RNG states are restored. At full resume only,
LGSA_RESUME_RUN and LGSA_SMOKE_RUN may be exported to this sbatch script, with
an explicit user-operated new submission. No automatic repeat jobs are created.
Do not point resume at a smoke or Experiment 1 checkpoint.

Outputs live under outputs/lgsa_goemotions/smoke/JOBID and seed42/JOBID.
Includes IDs, metadata, package versions, step/loss logs, checkpoints, 28-label
predictions, both threshold reports, per-label/support/TP/FP/FN, rare/nonrare
F1, source-labelled comparison CSVs, learned geometry, cosine heatmaps and PCA.
PCA is descriptive and does not prove orthogonality or causation. Single-seed
results cannot establish significance. Post-GAT orthogonality is a diagnostic,
not an enforced acceptance condition.

Baseline comparison references are copied small archived metric files, checked
against their recorded SHA256; no baseline trained weights are included.
