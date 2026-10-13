# Exp3B CPU-only recovery

This separate helper completes the v1 diagnostic that failed after all GPU
probes, because normalized Exp2 uses `rare_f1` while native Exp3 uses
`rare_mean_f1`. It does not modify either stored metric file or the v1 package.
Both aliases are explicitly supported; missing/conflicting aliases and any
numerical discrepancy still fail.

The helper verifies every original protected input against v1's recorded
pre-run hashes, verifies all four existing SUCCESS manifests, and reconciles
the saved GPU evidence: all 43,410 train IDs, 36,884 eligible terms, 2,714
forward batches, actual term/batch losses, sibling/co-GT exclusion, contribution
aggregates, 10 gradient batches, 12 BERT layers and unchanged-model audit.
It cannot recreate an absent pre-failure SUCCESS manifest: hashes of partial
artifacts are recorded at recovery start, and this limitation is explicit.

GPU is hidden with `CUDA_VISIBLE_DEVICES=''`. No model/checkpoint is loaded,
no new forward/autograd is run, and no optimizer is created. Original GPU
records are copied exactly into a new exclusive output directory:
`outputs/exp3b_diagnostics_seed42_recovered_v1`. The failed directory, original
experiments and checkpoints stay unchanged, including the original FAILURE.

The original independent diagnostic orchestration is reused in memory with
only a saved-probe provider and the strict metric-alias reconciler substituted.
No original source files are edited. Start/success/failure ntfy notifications
are included from the outset; delivery failure is nonfatal.

Run tests first, then the CPU-only audit, then completion. Always use Ubuntu's
existing `.venv-lgsa/bin/python`; do not install packages or retrain anything.

Local verification: 9 synthetic CPU compatibility/reuse tests, 19 existing
diagnostic tests, and 29 frozen Exp3 tests. These do not establish Ubuntu
scientific success. Only the subsequent Ubuntu audit and hash-verified SUCCESS
can establish successful completion.
