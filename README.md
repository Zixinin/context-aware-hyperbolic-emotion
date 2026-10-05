# context-aware-hyperbolic-emotion
FYP: Context-aware Multi-label Emotion Recognition with Hyperbolic Emotion Label Embeddings

This repository records the staged FYP experiments. Hyperbolic modeling is a later research stage; it has not been added to the current LGSA stability experiment.

| Experiment | Implementation | Current evidence |
|---|---|---|
| 1 — Standard BERT + GoEmotions | `experiments/exp1_bert_baseline/code/` | Original seed42 run and archived artifacts; raw bert-base-cased |
| 2 — LGSA + GoEmotions | `experiments/exp2_lgsa/code/` | Original seed42 run; documented parent-map patch and sentiment-derived hierarchy v0 |
| 2.1 — paired multi-seed stability | `experiments/exp2_1_multiseed/` | GPU job 3006384 stopped in BERT smoke raw-key verification; v3 compatibility fix prepared for audited rerun |

Experiment2.1 uses paired seeds42/123/2026, reuses seed42 only after audit, and preserves each model's existing optimizer, loss, hierarchy and training protocol. New runs independently initialize from raw pretrained BERT; they never inherit Experiment1 fine-tuned weights or smoke weights.

**Start here:** [Experiment2.1 instructions](experiments/exp2_1_multiseed/README.md) · [copyable HPC commands](experiments/exp2_1_multiseed/HPC_COMMANDS.md) · [repository layout](docs/REPOSITORY_LAYOUT.md).

## Version control and HPC

`code/` directories are byte-identical snapshots of the original owned source files, with hashes in `docs/source_snapshot_sha256.json`. The existing local folders `experiments/exp1_bert_baseline/goemotions/` and `experiments/exp2_lgsa/LGSA-GoEmotions/` retain their original files, backups and nested Git histories; the outer repository ignores them. No original run or frozen upstream checkout was modified during repository organization.

The official LGSA upstream is an external dependency pinned at `4ea3dfb39dcee82654f443811471dd29be32e87e`; the accepted HPC parent-map patch is `209f83762dae1b354ba40e68b6952482ecf96208`. Upstream source is not imported as an accidental nested Git link. The Experiment2 source includes its preparation scripts, adapter and change documentation.

Data, models, caches, archives, logs, native runs and generated reports are ignored. GitHub contains code and protocol documentation. Actual training runs on university Slurm GPU nodes.

After committing source changes, run `python3 tools/build_hpc_release.py`. It verifies the package against the current Git commit and creates a small code-only archive in `dist/`, with a checksum and a per-file source manifest. The package deploys next to the already validated HPC projects; it does not replace them or reinstall their environments.
