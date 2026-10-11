# Experiment 3: LGSA + DPT-derived peer hard-negative contrastive learning

Status: code preparation and local CPU unit tests only. Real Ubuntu LGSA regression,
GPU smoke and full seed42 remain pending. No improvement claim is made.

Read [implementation plan](docs/IMPLEMENTATION_PLAN.md) for audited tensors and
[official DPT provenance](upstream/SOURCE.md) for exact code sources and changes.
The existing Experiment 2 code/config/outputs are preserved byte-for-byte.

## What is reused

The official DPT `negative_sampling(..., mode='hard-verb')` function is copied
verbatim and actually executed. Its cosine positive-contrastive kernel is
adapted to one GT + its hardest non-GT sibling, using stable logsumexp.
This is a minimal DPT-derived module, not the complete DPT architecture.

The anchor is the MLM-transformed fine task MASK entering LGSA's existing
output decoder. Keys are that decoder's HGS/pre-GAT fine-label rows. Those
rows are frozen in Experiment 2, so the new term trains the existing anchor
path without unfreezing them. A temporary hook captures the actual decoder
input in the original forward. No extra encoder, GAT call, or random draw.

Defaults: seed42, peer_lambda0.5, temperature0.1, top_k1; all Exp2 hyperparameters
and threshold rules remain frozen. Every co-GT is excluded; neutral skips.
The disabled/zero-lambda path calls the original forward immediately.

## Ubuntu stage order

Commands below are templates: confirm the actual repository, existing `.venv-lgsa`
and Exp2 root/assets BEFORE executing them. Do not run model checks on the Mac.
Do not replace the existing environment or fabricate SLURM_JOB_ID.

From the confirmed outer FYP repository root, the typical runtime root is
`experiments/exp2_lgsa/LGSA-GoEmotions`. The launcher checks frozen source hashes,
dataset hashes, raw BERT revision/files, environment versions and actual GPU.
If an asset path still refers to HPC, supply `--model-path /absolute/path/to/the/pinned/snapshot`;
only the in-memory path changes, never old manifests.

If Ubuntu has raw BERT but no prepared JSONL data, first run this CPU-only stage
after transferring the complete Experiment3 package. It downloads the original
fixed dataset revision, not `main`, verifies each split's text, IDs, labels and
order against the hash-verified normalized seed42 baseline, checks historical
raw BERT file hashes, and publishes a new manifest only after all checks pass:

```bash
.venv-lgsa/bin/python experiments/exp3_dpt_peer/prepare_ubuntu_assets.py \
  --lgsa-root experiments/exp2_lgsa/LGSA-GoEmotions \
  --baseline-run experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42 \
  --pretrained-record environment/ubuntu/pretrained_assets.json
```

No existing data directory is overwritten. On failure, a new
`goemotions-v0.exp3-preparing` directory may remain for diagnosis; do not delete
or bypass it automatically. This stage performs no model construction, GPU
operations, test inference or training. Prepared test-data signature checks
are not test-based hyperparameter selection.

1. Save unit proof on Ubuntu:

```bash
.venv-lgsa/bin/python experiments/exp3_dpt_peer/unit_tests.py --out outputs/exp3_dpt_peer/unit_tests.json
```

2. Verify the copied official excerpt (network required, or use `--source-file`):

```bash
.venv-lgsa/bin/python experiments/exp3_dpt_peer/verify_upstream.py --out outputs/exp3_dpt_peer/dpt_upstream.json
```

3. Audit, then real lambda0 regression, then smoke. In each invocation substitute
`audit`, `regression`, or `smoke` for PHASE and use a new output directory:

```bash
PYTHONHASHSEED=42 .venv-lgsa/bin/python experiments/exp3_dpt_peer/launch.py \
  --phase PHASE \
  --lgsa-root experiments/exp2_lgsa/LGSA-GoEmotions \
  --unit-proof outputs/exp3_dpt_peer/unit_tests.json \
  --upstream-proof outputs/exp3_dpt_peer/dpt_upstream.json \
  --out outputs/exp3_dpt_peer/PHASE
```

Smoke additionally requires `--regression-run outputs/exp3_dpt_peer/regression`.
It uses1000 train/500 validation/1epoch, exercises the new gradient explicitly,
checks original backbone/GAT gradients and checkpoint reload, and does not read test.
Inspect losses, peer counts, selected_pairs, FAILURE.json, gradient probe and
SUCCESS.json before advancing. Smoke F1 is not a scientific result.

4. After passing/inspecting all gates, commit ONLY Experiment3 code on Ubuntu.
Record commit and package hashes. Do not commit datasets/checkpoints or existing
unrelated user changes. Then start a tmux session `fyp_exp3_seed42`. Within it:

```bash
mkdir -p logs
set -o pipefail
PYTHONHASHSEED=42 .venv-lgsa/bin/python experiments/exp3_dpt_peer/launch.py \
  --phase full \
  --lgsa-root experiments/exp2_lgsa/LGSA-GoEmotions \
  --unit-proof outputs/exp3_dpt_peer/unit_tests.json \
  --upstream-proof outputs/exp3_dpt_peer/dpt_upstream.json \
  --regression-run outputs/exp3_dpt_peer/regression \
  --smoke-run outputs/exp3_dpt_peer/smoke \
  --baseline-run /ABSOLUTE/VALIDATED/EXP2/SEED42/RUN \
  --out outputs/exp3_dpt_peer/seed42 \
  2>&1 | tee logs/exp3_seed42.log
```

The baseline may be the native `GOEMOTIONS_FULL_PASS` directory or the existing
normalized `NORMALIZED_RUN_PASS` seed42 directory. The latter requires all
local artifact hashes, frozen config/source/hierarchy, recorded historical
completion, checkpoint-reload evidence, four-epoch CSV, and exact dataset and
prediction ID/GT signatures. It does not claim a live Slurm or remote native
checkpoint check on Ubuntu. No marker is rewritten and no baseline is retrained.
Do not launch this template until its baseline path and stage evidence are verified.
Optional existing ntfy: set `FYP_NTFY_URL` in the Ubuntu shell; its value is not
printed/saved and notification errors do not fail training.

## Runtime and outputs

`launch.py` calls the frozen original `run.train` function. It adds one diagnostics
call after the original log-entry assignment through an AST transform; no original
model/trainer file is rewritten. It bypasses only the old main() Slurm wrapper and
obsolete comparison against BERT42, supplying the Ubuntu output path and stage gates.
The original MLM/classification/HGA losses, HGA ramp, optimizer, schedule,
checkpoint selection, split sampling and evaluation functions are reused.

Each output directory is created once and never overwritten. The first version
does not offer resume: a failed pilot stays preserved and a retry uses a fresh output
directory. It does not automatically tune, submit HPC jobs, or run additional seeds.

Full writes all original artifacts plus config/environment, raw verification,
training_log.csv, validation_metrics_by_epoch.csv, threshold_search.csv,
peer_training_diagnostics.csv, per_label_fixed_0.30.csv, exp2_vs_exp3_seed42.csv,
sibling_aggregate_comparison.json, the frozen12-row sibling_pair_comparison.csv,
six requested comparison/training figures and EXP3_SUMMARY.md. It reports margins
as probability differences and inversion as strict raw-logit ordering.
Analysis matches sample IDs and GT exactly and excludes all co-GT labels.
Checkpoint files retain original names checkpoint_best.pt/checkpoint_resume.pt
rather than moving to a new checkpoint/ layout; comparisons are in the new run
directory. These are output-layout deviations, not protocol changes.

SUCCESS.json is written only after training, checkpoint reload, reports, figures,
source/asset rechecks and artifact hashing complete. A single seed is a pilot;
interpret rare-label gains, aggregate discrimination, pair heterogeneity and
precision/recall together. A negative result is retained.
