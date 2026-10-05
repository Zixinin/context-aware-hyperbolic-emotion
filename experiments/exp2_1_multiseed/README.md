# Experiment 2.1 — paired multi-seed stability

**Status: code prepared; new HPC GPU runs have not been submitted or verified.**

Purpose: compare the existing Standard BERT and LGSA-GoEmotions protocols at the paired seeds **42, 123, 2026**, without a new modeling change. The complete user specification is in `docs/USER_EXPERIMENT_SPEC.md`.

## Execution order

1. Read-only BERT seed42 audit in the original `fypemotion` environment.
2. Read-only LGSA seed42 audit in the original `lgsa_goemo` environment; also verify patched WoS evidence.
3. Accept the prepared orchestration only after the audit; static checks.
4. One infrastructure smoke stage in a GPU allocation, covering both changed runner paths: each uses the existing 1000 training / 500 validation examples and one smoke epoch. Smoke never evaluates test data and never enters the scientific summary.
5. BERT full seeds123 and2026, plus seed42 only if its audited artifacts require rerunning.
6. LGSA full seeds123 and2026, plus seed42 only if required.
7. Inside the same GPU allocation, verify all six records, normalize artifacts, and produce both threshold comparisons, label analyses, figures and a preliminary `MULTISEED_SUMMARY.md`.
8. Once the sole GPU job and its batch/steps are **COMPLETED / 0:0**, run `status.py` to certify the report and mark the six records final.

The runtime order is gated. Preparing the code locally does not certify that either remote seed42 is reusable. Audit outputs explicitly distinguish `REUSED_VERIFIED`, `RERUN_REQUIRED` and an audit blocked by missing/current-incompatible infrastructure.

## Frozen protocols

| Setting | BERT | LGSA |
|---|---|---|
| Environment | fypemotion | lgsa_goemo |
| Torch | 2.6.0+cu118 | 2.5.1+cu118 |
| Transformers | 4.51.3 | 4.44.2 |
| Epochs / batch | 4 / 16 | 4 / 16 |
| Optimizer | AdamW, lr5e-5, wd0.01; original bias/LayerNorm exclusions | Adam, lr1e-4, wd0 |
| Warmup | original ceil:1086 | original floor:1085 |
| Token budget | 50 | text50 + prompt4 =54 |
| Precision | FP32; TF32 off | FP32; original Torch TF32 defaults |
| Determinism | original deterministic-algorithms=True | original False; cudnn deterministic=True; scatter/GAT may be nondeterministic |
| Evaluation batch | 64 | 16 |

Both use raw `google-bert/bert-base-cased` revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`, official splits43410/5426/5427, the frozen seven rare labels and 28 fine-label evaluation. `model_revision=main` remains in the legacy BERT config to preserve its exact bytes; loading uses the immutable resolved revision in its asset manifest with network access disabled.

Only seed differs within a model family. LGSA's legacy config `experiment_name` contains `seed42`; it is kept as an immutable legacy label. Actual model/seed/job/run identity is explicit in new metadata and directory names.

No original code is edited. `original_lgsa` must remain at `4ea3dfb39dcee82654f443811471dd29be32e87e`; the HPC working patch must remain at `209f83762dae1b354ba40e68b6952482ecf96208`.

## Independence and parameter verification

- Every phase runs in a fresh Python process using its own original environment.
- Before training, every backbone tensor is compared directly against the pinned raw safetensors file. Full-model initialization is never loaded from a trained checkpoint or from smoke.
- For LGSA, an initialization-only process executes the original construction and HGS path, writes raw backbone/full-model fingerprints, and exits **before optimizer construction or training**. A separate full process reconstructs the model from raw weights and must match that same-seed reference. This is not another per-seed training smoke.
- Original seed42 BERT did not record an initial tensor digest. The reuse audit relies on its complete original source hashes, raw-asset hashes, independent raw-loading code path, immutable predictions and completed-run evidence; it does not fabricate a historical fingerprint.
- No cross-seed resume. An incomplete new run is preserved for diagnosis; this release does not automatically resume or repeat failed jobs.

## Deployment

See [HPC_COMMANDS.md](HPC_COMMANDS.md) for exact upload, audit, submit and inspection commands.

The HPC release directory must be a sibling of the **existing** two HPC projects:

```text
/gpfs/work/aac/zixinfeng23/fyp/
├── goemotions/                 # original environment/assets/source/seed42
├── LGSA-GoEmotions/            # original environment/assets/source/seed42
└── FYP-Experiment-2.1-MultiSeed/
```

The Git repository's local layout is for version control. The release builder packs this directory under the HPC name above; do not move the existing HPC projects to match the local Git layout.

All required assets already exist on HPC. The release does not install dependencies or download models/data. Any source/environment/data drift stops the workflow. Missing scheduler evidence also stops the audit instead of wasting GPU time on an unjustified rerun.

## Outputs

```text
runs/job<GPU_JOB>/
  bert_infrastructure_smoke/
  lgsa_infrastructure_smoke/
  bert_seed123/ ...
  lgsa_seed123/ ...
  ledger.json
outputs/job<GPU_JOB>/
  bert_baseline/seed42/ seed123/ seed2026/
  lgsa_goemotions/seed42/ seed123/ seed2026/
  multiseed_comparison/
    overall_summary.csv
    multiseed_mean_std.csv
    paired_seed_deltas.csv
    paired_delta_stability.csv
    rare_label_seed_stability.csv
    grief_relief_seed_analysis.csv
    per_label_multiseed_comparison.csv
    per_label_multiseed_comparison_by_wins.csv
    improved_label_counts_by_seed.csv
    label_win_distribution.csv
    precision_recall_tradeoff.csv
    validation_curves_all_seeds.csv
    threshold_stability.csv
    threshold_ranking_comparison.csv
    figures/                   # PNG + PDF; both threshold settings
    MULTISEED_SUMMARY.md
    PRELIMINARY.json
    INTEGRITY_PRELIMINARY.json
    INTEGRITY_PASS.json
    SUCCESS.json
```

Normalized seed42 views are newly created from existing logits; original artifacts remain immutable. Checkpoint files are referenced by absolute path + SHA256 instead of duplicated. Metrics use proportions in CSV; the Markdown report shows percentages and percentage-point differences. Standard deviations use `ddof=1`. No statistical significance test or causal attribution to an individual LGSA component is performed.

## Verification boundary

Local tests cover metrics/counts, exact historical BERT prediction regression, seed propagation, raw-weight verification, failure gates and complete synthetic report generation. They do not validate RTX4090 training or current HPC files. Those checks run on HPC in the gated order above. A preliminary report inside a still-running job is not final scheduler completion: use `status.py` after the GPU job ends.

If the GPU job fails, no final result is certified. Because the specification requires the containing Slurm job to finish successfully, earlier phases inside a failed allocation are not silently reused. Inspect the cause before preparing a separate retry.
