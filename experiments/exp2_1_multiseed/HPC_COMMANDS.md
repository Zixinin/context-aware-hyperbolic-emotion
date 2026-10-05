# Exact commands — user-operated HPC

The new GPU experiments have not been run by Codex. Run these commands in order. Do not delete or change either original HPC project.

## 1. Mac: build and upload

The source must first be committed to the local Git repository. A committed release records that exact commit and per-file hashes. Generated release files are ignored by Git.

```bash
cd '/Users/fengzixin/Desktop/PSP303-FYP/context-aware-hyperbolic-emotion'
python3 tools/build_hpc_release.py
rsync -avP dist/fyp21_hpc.tar.gz dist/fyp21_hpc.tar.gz.sha256 \
  zixinfeng23@login.hpc.xjtlu.edu.cn:/gpfs/work/aac/zixinfeng23/fyp/
```

`rsync -P` retains a partial transfer. Re-run the same upload command if disconnected; the archive checksum is verified before extraction. No data/model downloads are required.

## 2. HPC: install the new directory and audit seed42

Log in from the Mac with `ssh zixinfeng23@login.hpc.xjtlu.edu.cn`, then run:

```bash
(
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp
sha256sum -c fyp21_hpc.tar.gz.sha256
test ! -e FYP-Experiment-2.1-MultiSeed || {
  echo 'Directory already exists: stop to avoid overwriting code or run evidence.' >&2
  exit 1
}
tar -xzf fyp21_hpc.tar.gz
cd FYP-Experiment-2.1-MultiSeed
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u audit_seed42.py
)
```

This is read-only hashing, metadata and saved-prediction/data checking; it performs no model forward/backward. If your site requires even these CPU checks to be scheduled, run the audit command inside an ordinary CPU allocation. It does not need a GPU.

Expected final marker: `SEED42_AUDIT_PASS`. It will print each model's `REUSED_VERIFIED` or `RERUN_REQUIRED` and the exact training plan. An unavailable environment, source drift, or missing scheduler history is an audit error; paste the exact error instead of bypassing the gate.

If the directory is already installed and an audit merely needs to be retried, use:

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u audit_seed42.py
```

## 3. HPC: submit only after the audit passes

```bash
(
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u submit.py
)
```

This validates both batch scripts with `sbatch --test-only`, submits **one gpu4090 job** (1GPU,4CPU,20GB, maximum6h), then a dependent **cpu8358 reporting job** (2CPU,8GB, maximum1h). The wall-time requests are limits, not predictions of actual training duration. Normal plan: two smoke paths, four new full runs, reuse both seed42 records. If a seed42 audit finds an actual mismatch, only that model's seed42 is added.

Notifications go to the existing school email: training BEGIN/END/FAIL, reporting END/FAIL. The report job starts only after the GPU job succeeds. Failed dependencies cancel reporting. Do not interpret an END email alone as proof of scientific validity.

## 4. HPC: check queue, accounting and results

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u status.py
```

During training, each phase has its own log under `runs/job<GPU_JOB>/`. For example:

```bash
tail -n 60 runs/job<GPU_JOB>/bert_seed123.log
```

Replace `<GPU_JOB>` with the actual numeric ID printed at submission. After completion `status.py` prints `EXPERIMENT_2_1_COMPLETE_VERIFIED` only when both jobs finished COMPLETED/0:0 and report hashes pass. It then displays the final Markdown report.

## If only report submission fails

`submission.json` records the GPU job immediately. **Do not re-submit training.** Inspect that file and the scheduler rejection; after resolving CPU partition/account eligibility, use `retry_report.py` to submit the report alone. The exact error is needed if a partition/QOS change is necessary.

## GitHub

Code is in the local Git repository. HPC `runs/`, `outputs/`, data and weights are excluded. Commit source changes before rebuilding a release; do not upload a different release over a running experiment. GitHub publishing is separate from Slurm submission.
