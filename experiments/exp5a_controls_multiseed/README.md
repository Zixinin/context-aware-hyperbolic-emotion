# Exp5A.2 — CLS vs Mean: new paired seeds123/2026, frozen seed42 reuse

Scope: train exactly four new models, independently from pinned raw BERT:
CLS123, Mean123, CLS2026, Mean2026. Do not retrain seed42 or Query; do not implement
prototypes, new losses/hierarchy/geometry or extra seeds. Results stay in
`results/exp5a_controls_multiseed/run_*`.

Both prior source packages are read-only dependencies pinned in
`frozen_dependency.json`. Their completed seed42 three-way report and all model
artifact hashes are checked, including saved-logit metric recomputation. No
source/result mutation or new GPU forward is performed for historical models.

The new trainer preserves the seed42 epoch/gradient/optimizer/checkpoint loop
byte-for-byte (tested). Model, optimizer, tokenizer, collator, metrics and split
loader come from the frozen infrastructure. Only seed, independent output paths
and provenance/status routing differ. Both new seeds use raw last-layer CLS or
unmasked-token mean (including CLS/SEP, excluding PAD), no pooler, identical
classifier initialization and per-label independent dropout. Four epochs,
batch16, max length50, AdamW5e-5/weight_decay0.01, linear warmup0.1, BCE,
FP32/eager/deterministic, fixed0.30 evaluation and best validation Macro-F1.

Each seed sets its own Python hash, Python/NumPy/Torch/CUDA seed and loader shuffle
generator. The initial discarded random draw preserves the head/RNG convention
used by seed42. Target CPU preflight checks all197 raw encoder tensors and actual
seedwise CLS/Mean/query-initialization head and RNG parity. Instantiating a query
model in CPU preflight is only an initialization oracle, NOT Query training or a
new Query result.

## User-operated launch

1. Transfer ZIP/SHA to Ubuntu project root. Verify and extract without overwrite.
2. `bash experiments/exp5a_controls_multiseed/launch.sh notify-test`; confirm phone.
3. `bash experiments/exp5a_controls_multiseed/launch.sh audit`; review emitted root.
4. In tmux, `bash experiments/exp5a_controls_multiseed/launch.sh chain /absolute/root`.

Chain: fresh target audit -> all four smoke runs -> all four formal runs serially
-> six-model/three-paired-seed summary -> artifact verification -> stop.
Smoke uses each run's own seed,1000train/500validation/1epoch/63steps; no test
predictions. Full starts fresh from raw weights, not smoke/seed42 weights; each
model must pass4epochs/10856steps, gradient checks and numerical reload checks.
Test is evaluated once per new full model after validation-only checkpoint and
secondary threshold selection. CPU verification may read saved test predictions.

Notifications are wired from first launch: phase/model start, verification success
and failure. Bounded nonfatal delivery preserves original exit code. Failed
partial directories are retained, never overwritten/deleted/resumed silently.
Completed hash-verified models may be skipped on rerun. If only comparison fails,
retain GPU outputs and request CPU-only report recovery; do not retrain.

## Summary and limits

`EXP5A_CLS_MEAN_MULTISEED_SUMMARY.md` reports model mean±sampleSD (ddof1), every
per-seed metric, paired Mean−CLS delta mean±sampleSD and sign counts, per-label
means/SD and consistency, grief+relief signed contribution per seed and in the
mean, checkpoint validation diagnostics and parameter counts. CSV/JSON files
retain unrounded values. Statistics average independently evaluated per-seed
metrics, not ensemble predictions. All three seeds are included regardless of
outcome. Ratios with zero net change are undefined, not zero.

`MULTISEED_SUCCESS.json` records report hashes, original seed42 success identities
and new model proofs. Final marker: `EXP5A_CLS_MEAN_MULTISEED_COMPLETE_VERIFIED`.
To verify later: `bash experiments/exp5a_controls_multiseed/launch.sh verify /root`.

Three paired seeds give limited repeatability evidence, not significance,
equivalence or universal improvement. Query remains single-seed. Validation/test
rankings can differ; do not change training/thresholds/checkpoint choices based
on observed test scores. No automatic future experiment is launched.
