# Repository organization

```text
context-aware-hyperbolic-emotion/
├── experiments/
│   ├── exp1_bert_baseline/
│   │   ├── code/                   # versioned original BERT source snapshot
│   │   └── goemotions/             # preserved local project; ignored
│   ├── exp2_lgsa/
│   │   ├── code/                   # versioned adapter, scripts and documentation
│   │   └── LGSA-GoEmotions/        # preserved local projects/git histories; ignored
│   └── exp2_1_multiseed/           # latest stability experiment code
├── tools/build_hpc_release.py
├── docs/source_snapshot_sha256.json
├── results/                       # reserve for intentionally curated summaries
├── 数据集/                         # existing local dataset; ignored
└── dist/                          # generated upload archive; ignored
```

The snapshots avoid two Git problems in the moved original projects: nested `.git` directories would otherwise become unhelpful gitlinks, and LGSA's original `.gitignore` hides the independently versioned adapter. The original project directories and histories stay intact; the clean snapshots contain their actual source files as normal Git files.

The multi-seed deployment continues to use the original flat HPC paths. Do not overwrite HPC source files with the new repository layout. Experiment2.1 calls the original hash-verified training functions through small orchestration wrappers.

The root Git remote was already configured as `git@github.com:Zixinin/context-aware-hyperbolic-emotion.git` when this organization was performed. Repository organization is local; publishing is a separate user action.

No model weights, official datasets, archived trained checkpoints, private credentials, or synthetic test outputs belong in the Git index. Original scientific artifacts remain on HPC and in the existing local backups.
