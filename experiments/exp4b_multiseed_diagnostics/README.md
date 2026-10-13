# Exp4B multi-seed extension and read-only diagnostics

User authorization: reuse the completed seed42; train only123/2026 with identical
full hierarchy and protocol; fixed0.30 diagnostics; stop after final summary.
No dataset, original Exp2 source, Exp4A artifact, pilot source or completed pilot
artifact is edited. No pruning, flattening, threshold tuning, peer or hyperbolic.

The15-file validated pilot is hash-pinned. New seed orchestration is derived from
its main function by five checked changes: two seed metadata values, matched
Exp2 path, report title and truthful full-success status. The seed-specific
environment guard uses the actual PYTHONHASHSEED; it is not spoofed. All model,
forward/backward, sampling, optimizer, schedule and loss code is unchanged.
The pilot training-loop derivation still has exactly its original five hierarchy
dimension edits. Reversing the seed orchestration changes reproduces the pilot
AST. Each new seed has its own audit, fresh4-update smoke and raw-reset full run.
The original pilot's historical seed42 authorization file remains immutable; this
separate protocol explicitly authorizes the two new seeds.

Each baseline is the normalized completed Exp2 result of the SAME seed from
job3006610. All original provenance, checkpoint, dataset signature, prediction
hash, configuration and metric recomputation checks remain, with only the seed
guard and expected config seed derived. Only config.seed changes from Exp4B42.
The frozen legacy auxiliary validation-selected-threshold report is retained for
loop identity, but never used for this analysis or tuning. The report uses0.30 only.

Sequential execution: pilot37 units -> extension units -> verify all three Exp2
baselines and completed Exp4B42 ->123 audit/smoke/full ->2026 audit/smoke/full ->
read-only diagnostics -> verify files -> completion marker. Failure retains all
outputs. Resume skips hash-verified completed stages; incomplete stage directories
are refused, not deleted/retrained silently. Diagnose those before recovery.
Notifications have at most3 bounded attempts and do not override experiment exits.

Diagnostics export raw numeric CSV/JSON intermediates and Markdown, not an Excel
workbook. One label-seed record per row;84 rows plus28 label-level mean/std rows.
Standard deviation uses ddof1. Ratios use Exp4B/Exp2; zero denominator is null/blank,
not0. Depth is ROOT0 and edge count; node path length is depth+1. Rare definition
is the frozen7 least frequent TRAIN labels. TRAIN support is multi-label occurrence
count over the full official43410 examples, not raw annotator counts.

Correlations are descriptive Pearson/Spearman perseed28 labels and acrossseed
label means28 labels, with an extra27-label sensitivity excluding neutral. No
pseudo-independent84-label significance claims. Frequency-depth correlation is
reported as a confounding warning. P/R contributions to F1 use a symmetric
algebraic decomposition, not a causal experiment.

Read saved steps.jsonl: classification, MLM, raw HGA, weighted HGA, combined
auxiliary and total. Validate the original ramp and loss sum. Classification
supervises all56 hierarchy nodes; a fine28-only loss was not logged separately.
Epoch means average optimizer updates, not examples. Missing loss components are
explicitly unavailable. Diagnostics never invoke a model forward or training.

Final report: diagnostics/EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY.md.
Completion marker: EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE.
This source delivery does not assert that the new GPU runs are already complete.
