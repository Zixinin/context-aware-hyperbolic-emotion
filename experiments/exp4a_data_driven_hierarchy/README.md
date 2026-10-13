# Exp4A TRAIN-only hierarchy

Run on D3020-A3 using .venv-analysis (Python 3.12.3), in tmux.
No classifier, GPU, Slurm, or Exp4B is invoked. Existing experiment files are not modified.

Dependencies: existing NumPy 1.26.4 and matplotlib 3.9.2, plus SciPy 1.14.1 and scikit-learn 1.5.2. pandas is not needed. Do not upgrade unrelated environments.

Audit gate (use a fresh separate output directory):

```bash
.venv-analysis/bin/python experiments/exp4a_data_driven_hierarchy/run_exp4a.py --audit-only --output outputs/exp4a_data_audit
```

After the audit passes, from tmux:

```bash
bash experiments/exp4a_data_driven_hierarchy/launch.sh
```

The formal output directory is outputs/exp4a_data_driven_hierarchy. Nonempty output directories and existing formal logs are refused. Failed runs retain partial outputs; to retry, use a new --output path and a new log, preserving the old ones.

Reopen and verify every manifested artifact:

```bash
.venv-analysis/bin/python experiments/exp4a_data_driven_hierarchy/run_exp4a.py --verify-only
```

Bootstrap is fixed at B=200, seed=2026, resampling TRAIN comments. Partition cuts use cut_tree for exactly k=2..10. Tree support is exact-clade recurrence. Full emotion root support is trivial and excluded from reported stable clades. support70 annotates; it does not prune.

The supplied protocol retains all rater rows in PRIMARY. Official GoEmotions code excludes no-label rows; a separate sensitivity tree applies that filter. Its results do not select the primary hierarchy. Reference sources and the Ward/correlation caveat are in PAPER_METHOD_AUDIT.md.

Exp4B readiness uses a conservative declared heuristic (one singleton-free k with mean ARI and NMI >=0.80 and largest cluster <=20/27). A clear candidate k requires the same eligible winner in silhouette and mean ARI, with margins >=0.02 and >=0.05 respectively, and at least two eligible candidates. Otherwise no unique k is supported. READY_WITH_CAUTION requires reviewing weak clades and robustness; otherwise NOT_STABLE_ENOUGH. These are design heuristics, not statistical hypothesis tests. No claim comparing raw and simplified bootstrap stability is made because only PRIMARY is bootstrapped.

SUCCESS.json means computational acceptance passes; visual review and scientific interpretation remain part of handoff. ntfy start/success/failure notifications are enabled, timeout-bounded, and do not determine scientific validity.

Git staging must explicitly name only this experiment directory and any selected small result summaries. Do not stage unrelated modifications, raw files, matrices, outputs, environments or caches.
