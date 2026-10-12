# Exp4B: full train-only human-judgment hierarchy in unchanged LGSA

Authorized scope: hierarchy validation → independent GPU smoke → fresh seed42
full training → paired comparison to normalized Exp2 LGSA seed42/job3006610.
No123/2026 run, pruning, k4 flattening, peer loss, hyperbolic module or new loss.

## Locked controls and authorized deterministic changes

Canonical GoEmotions train/dev/test remain43410/5426/5427, text/IDs/GT/order verified
against the normalized baseline AND local canonical TSVs. All28 original fine IDs
remain0..27. Fixed primary threshold0.30 and best validation Macro-F1 checkpoint.
Same raw bert-base-cased revision, tokenizer, float32, Adam1e-4, batch16, four epochs,
linear schedule,0.1 warmup, clip1, unchanged original HGS/GAT/MLM/classification/HGA.
The original validation-only threshold grid is retained as secondary reporting;
it does not select a hierarchy or replace the primary fixed-threshold comparison.

Full-tree SHA256 is pinned. All26 binary clusters, including8 unstable clusters,
EMOTION and NEUTRAL stay intact; logical ROOT is excluded from supervision IDs.
Names are preserved, not semantically renamed. Auxiliary IDs are deterministic:
EMOTION28, NEUTRAL29, raw_cluster_001..026 become30..55.
The unchanged semantic initializer tokenizes the literal internal node IDs; it
does not invent emotion names or replace them with descendant-embedding means.

| Field | Old | New | Interpretation |
|---|---:|---:|---|
| Final emotion classifier output |28|28| Unchanged task labels |
| Auxiliary hierarchy supervision nodes |32|56| Derived from full tree |
| Internal nodes excluding ROOT |4|28| Derived from full tree |
| Modeled depth |2|9| Derived from full tree |
| Text window |50|50| Unchanged CLS/text/SEP tokenization budget |
| Total input length |54|68| text_window +2*depth, hierarchy-induced |

Upstream Prompt.num_labels is the number of ALL auxiliary hierarchy nodes, not
the public emotion classifier dimension. `EmotionClassifier.forward` returns
exactly Bx28, taking each original leaf's logit from its actual depth. There is no
new learned head. ROOT/EMOTION/NEUTRAL/clusters never enter exported task predictions
or emotion metrics. The original auxiliary objective still supervises ancestors,
as old Exp2 already did for its4 coarse nodes. Do not falsely set upstream
num_labels=28: that would drop internal supervision and change the mechanism.

## Reuse boundary

All25 Exp2 adapter files and four model source files are hash-checked and never
edited. The original `train` function is compiled in a private namespace with
exactly five checked dimension literal edits:32→56 model supervision classes,
graph max33→64, embedding extension35→66, two label slices[-35:-3]→[-66:-10].
Running this adapter with32 nodes/two depths yields EXACTLY the original train AST.
Optimizer, loops, MLM and categorical loss, HGA formula/weights, schedule, RNG
handling, best checkpoint/reload logic and durable200-step checkpoints remain.
Separate helpers generalize tree targets, input prompts, fine logit selection and
HGS acceptance, and add weighted F1 to reporting only. Original GAT source and
classification/HGA loss implementations are unchanged.

The small smoke uses64 TRAIN/32 validation examples, four optimizer steps and one
validation pass. It includes original validation threshold/reload diagnostics but
never test inference. Full starts in a NEW process with independently checked199
raw BERT tensors and50 legacy LayerNorm aliases. Its initial backbone/full model
digests must match fresh smoke initialization, never smoke-trained weights.

## Execution and gates

Install into `experiments/exp4b_data_driven_hierarchy` without overwriting files.
Use the exact Ubuntu project `.venv-lgsa/bin/python`, no environment upgrades.

```bash
bash experiments/exp4b_data_driven_hierarchy/launch.sh smoke
```

Runs Ubuntu unit tests → source/data/baseline/config audit → real GPU smoke.
Outputs only new `results/exp4b/run_<UTC timestamp>/`. Start/success/failure ntfy
notifications are bounded/nonfatal; pipefail preserves training/log failures.
There is NO automatic full launch. Keep the printed EXP4B_RUN_DIR.

After reviewing the smoke gate, commit ONLY the new Exp4B code. Preserve unrelated
dirty/untracked files. The full launcher checks committed, unchanged Exp4B sources.

```bash
bash experiments/exp4b_data_driven_hierarchy/launch.sh full /absolute/verified/run_directory
```

No previous output directory is overwritten. The original trainer keeps durable
checkpoint_resume.pt every200 updates. If interrupted, retain everything and
inspect evidence before authorizing an explicit same-run recovery; do not blindly
restart, overwrite, or inherit another run's model/optimizer state.

A failure in baseline/data verification stops before training. The normalized
baseline records historical HPC completion; this launcher does NOT connect to HPC,
Slurm, or SSH. It also does not retrain the old baseline on Ubuntu.

## Outputs and interpretation

audit/: hierarchy_validation.json, exp4b_config_comparison.json, baseline proof,
source/env provenance, dimension-only train diff, SUCCESS.json.
smoke/: smoke_test_report.json, independent raw proof, gradients, HGS/reload proof,
finite4-step log, N×28 validation predictions, SUCCESS.json.
seed42/: four-epoch logs/checkpoints, N×28 validation/test predictions, fixed0.30
micro/macro precision/recall/F1, weighted F1, secondary original-grid reports,
labelwise_metrics.csv, exp2_vs_exp4b_seed42.csv, exp2_vs_exp4b_labelwise.csv,
six stable-pair descriptive summaries, EXP4B_SUMMARY.md and hash manifest.

Pilot completion marker: EXP4B_SEED42_PILOT_COMPLETE_VERIFIED.
Do NOT claim EXPERIMENT_4B_COMPLETE_VERIFIED or three-seed consistency yet.
Do NOT change hierarchy after seeing metrics. Neither gains nor losses trigger an
automatic redesign or ablation. This is full-hierarchy replacement, not edge-only
causal attribution: extra ancestor targets/nodes/prompts follow the tree.
Old HPC and new Ubuntu4090D hardware/runtime are not identical; disclose this even
when frozen versions/source/protocol match. One seed has no significance claim.

Mac checks are synthetic CPU checks only. Actual frozen BERT/GAT forward/backward
is required on Ubuntu before any formal result can be asserted.

## TSV parser repair v1 (2026-10-12)

The first Ubuntu attempt passed25 units, then stopped BEFORE GPU smoke during
canonical TSV comparison. The checker incorrectly used QUOTE_NONE, retaining
enclosing/doubled quotes in some text fields. QUOTE_MINIMAL decoding reproduces
the frozen Exp2 full-row, ID and gold signatures in all three splits exactly.
This repair changes the read-only checker, not any dataset file, training text,
split/order, tokenizer, hierarchy or model/loss. It adds canonical byte SHA pins
and exact decoded-record signature checks; no assertion is removed or bypassed.
Two added regression tests cover quoted fields, byte preservation and all three
historical signatures. The repaired package must rerun all27 Ubuntu units and
preflight before a NEW smoke attempt. Keep the failed run and original logs.
## Raw backbone checker repair v2 (2026-10-12)

The parser-v1 Ubuntu rerun passed all27 CPU tests and the complete audit,
then stopped before HGS/CUDA/optimizer on an incorrect199-tensor assertion.
Original LGSA Prompt explicitly uses BertModel(add_pooling_layer=False), which
has197 backbone tensors and50 legacy LayerNorm aliases. Standard BERT with a
pooler has199; adding a pooler to LGSA would change the frozen architecture.
The checker now requires the exact197-key encoder/embedding schema, no pooler,
the original50 aliases and unchanged SHA/dtype/shape/exact-value verification.
Ten added regression tests cover this contract, including negative cases.
No training loop, model implementation, hierarchy or configuration is changed.
The previous audit PASS remains historical evidence, not a compatible gate for
new code: rerun the complete smoke batch with fresh package hashes.
