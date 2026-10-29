# Exp5A-Control: raw CLS vs masked mean vs frozen label-query

Only independently train `cls_seed42` and `mean_seed42`, four epochs each, serially
on D3020-A3 / RTX 4090 / `.venv-bert/bin/python`. Reuse C from
`results/exp5a/run_20261013T093754Z/seed42`. No extra seeds, new methods or Exp5B.

## Frozen dependency and fairness

This package depends on the already-deployed Exp5A package. Its source and package
manifest are pinned in `frozen_dependency.json`. Never edit those sources/results.
Import reuse is explicit: data, collator, shuffle loader, seed setup, metrics,
optimizer, prediction and split-saving functions come from that frozen package.
The controls have independent model/training/verifier/report entrypoints.

Every hyperparameter is read from the pinned Exp5A config. A uses raw last-layer
CLS (no learned tanh pooler); B averages unmasked tokens including CLS and SEP,
excluding PAD. Historical Exp1 uses a pooler and is reference only.

The classifier's weights/bias and `[B,28,D]` independent-per-label dropout match C.
At evaluation this is exactly `Linear(D,28)` for A/B. Expanding the pooled vector
for dropout is not token attention or an additional module. Applying ordinary
shared `[B,D]` dropout would change another aspect of C's training, so is avoided.
One temporary discarded normal draw (no query parameter/module) matches C's
classifier initialization and post-initialization RNG state. Target CPU audit
verifies all 197 raw encoder tensors and actual C head/CPU/CUDA RNG parity.

CLS/mean: 107741212 trainable parameters each; C: 107762716, exactly 21504 more.
The only learned architectural change relative to C is the representation
mechanism (including absence of query parameters). Independent training naturally
creates different optimization trajectories. No new attention visualizations or
semantic claim is added to A/B.

## Gates and launch

1. Verify ZIP SHA and package manifest; `bash -n launch.sh`.
2. `bash experiments/exp5a_controls/launch.sh notify-test`; confirm phone receipt.
3. `bash experiments/exp5a_controls/launch.sh audit`; record emitted absolute root.
4. `bash experiments/exp5a_controls/launch.sh smoke /absolute/root` in tmux.
5. Review both synthetic/CPU/audit/smoke proofs; no smoke test-set predictions.
6. `bash experiments/exp5a_controls/launch.sh full /absolute/root` in tmux.
7. `bash experiments/exp5a_controls/launch.sh verify /absolute/root`.

Launch includes start/success/failure ntfy notifications and preserves the original
exit code even when notifications fail. Shell exports deterministic CuBLAS and
Python hash settings, offline flags and clears conflicting Python environment.
Full training refuses to start unless BOTH smoke runs pass. Target preflight is
rerun read-only immediately before each phase; package/pip/C changes are blockers.
Never run real BERT training on the Mac.

Completed, hash-verified models may be reused by rerunning a phase. Existing
partial model directories are NOT resumed/overwritten: preserve them and ask for
recovery guidance. No automatic deletion. A/B model failures save metadata; a
comparison failure retains both models for CPU-only recovery.

## Evaluation and evidence

Best validation Macro-F1 at fixed 0.30 selects the checkpoint (earlier epoch wins
ties). Reload is numerically checked on validation. Test predictions are produced
once per new control after selection; verification only recomputes metrics on CPU
from saved logits. Secondary global threshold uses validation only, identical to
C, and is not the formal comparison. Seven rare IDs remain 16,21,23,19,12,24,14.

The independent verifier checks target provenance, canonical IDs/gold labels,
config, steps (full 10856 / smoke 63), checkpoint/history, threshold grid, metric
recomputation, per-label supports, rare definition, artifacts and source hashes.
Current package versions and pip freeze must match frozen C exactly. C's existing
verifier plus all saved artifact hashes and prediction metrics are checked.

Final root artifacts:

- `EXP5A_CONTROL_SUMMARY.md`: historical reference + three-way formal table,
  paired deltas in pp, best-epoch/validation/parameter diagnostics, all28 labels,
  grief+relief contribution to Macro/Rare, scientific boundaries.
- `CONTROL_COMPARISON.json`: unrounded machine-readable metrics/deltas/provenance.
- `CONTROL_PER_LABEL_COMPARISON.csv`: all28 paired per-label F1/support changes.
- `CONTROL_SUCCESS.json`: comparison hashes, emitted only after all gates pass.

Final identity: `EXP5A_CONTROL_COMPLETE_VERIFIED`. Notifications alone are not
proof. Single seed cannot establish significance, equivalence or consistency;
the report intentionally does not invent a post-hoc 'clear/close' cutoff. Use
signed grief+relief contributions; ratios may exceed 100% when gains offset.
No further experiment is launched after completion.

Earlier attention diagnostic: 128 validation texts / 3584 label decisions,
not 3584 validation examples. Inference equivalence is not training equivalence.
