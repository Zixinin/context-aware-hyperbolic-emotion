# Exact commands — user-operated HPC

The new GPU experiments have not been run by Codex. Run these commands in order. Do not delete or change either original HPC project.

## 1. Mac: build and upload

The source must first be committed to the local Git repository. A committed release records that exact commit and per-file hashes. Generated release files are ignored by Git.

```bash
cd '/Users/fengzixin/Desktop/PSP303-FYP/context-aware-hyperbolic-emotion'
python3 tools/build_hpc_release.py --archive-name fyp21_hpc_layernorm_v3.tar.gz
rsync -avP dist/fyp21_hpc_layernorm_v3.tar.gz dist/fyp21_hpc_layernorm_v3.tar.gz.sha256 \
  zixinfeng23@login.hpc.xjtlu.edu.cn:/gpfs/work/aac/zixinfeng23/fyp/
```

`rsync -P` retains a partial transfer. Re-run the same upload command if disconnected; the archive checksum is verified before extraction. No data/model downloads are required.

## 2. HPC: preserve the failed GPU-only release, install v3 and audit seed42

Log in from the Mac with `ssh zixinfeng23@login.hpc.xjtlu.edu.cn`, then run:

```bash
(
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp
sha256sum -c fyp21_hpc_layernorm_v3.tar.gz.sha256
test ! -e FYP-Experiment-2.1-MultiSeed-failed-3006384
test -e FYP-Experiment-2.1-MultiSeed
test "$(/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -c 'import json; print(json.load(open("FYP-Experiment-2.1-MultiSeed/submission.json"))["train_job"])')" = 3006384
test "$(sacct -j 3006384 --noheader --parsable2 --format=JobIDRaw,State,ExitCode | awk -F'|' '$1=="3006384" {print $2 "|" $3}')" = 'FAILED|1:0'
mv FYP-Experiment-2.1-MultiSeed FYP-Experiment-2.1-MultiSeed-failed-3006384
tar -xzf fyp21_hpc_layernorm_v3.tar.gz
cd FYP-Experiment-2.1-MultiSeed
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u audit_seed42.py
)
```

The failed package and `runs/job3006384/` remain preserved under `-failed-3006384`; the earlier `-cpu-report-v1` backup remains untouched. The failed job was a BERT smoke setup failure, with no full runs. The new audit checks existing artifacts and does no model training.

Expected final marker: `SEED42_AUDIT_PASS`. It will print each model's `REUSED_VERIFIED` or `RERUN_REQUIRED` and the exact training plan. An unavailable environment, source drift, or missing scheduler history is an audit error; paste the exact error instead of bypassing the gate.

Before submitting v3, verify the actual cached raw BERT weights using the original BERT environment, with no GPU training:

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/gpfs/work/aac/zixinfeng23/.conda/envs/fypemotion/bin/python -u preflight_raw.py
```

The required marker is `RAW_CASED_CPU_PREFLIGHT_PASS`; it reports the number of exact backbone tensor comparisons and legacy LayerNorm keys. A mismatch stops before another GPU request.

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

This validates the single batch script with `sbatch --test-only`, then submits **one gpu4090 job** (1GPU,4CPU,20GB, maximum8h) that includes smoke, training and report generation. The wall-time request is a limit, not a prediction of actual training duration. Normal plan: two smoke paths, four new full runs, reuse both seed42 records. If a seed42 audit finds an actual mismatch, only that model's seed42 is added.

Notifications go to the existing school email: GPU job BEGIN/END/FAIL. Do not interpret an END email alone as proof of scientific validity.

## 4. HPC: check queue, accounting and results

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u status.py
```

During training, each phase has its own log under `runs/job<GPU_JOB>/`. For example:

```bash
tail -n 60 runs/job<GPU_JOB>/bert_seed123.log
```

Replace `<GPU_JOB>` with the actual numeric ID printed at submission. After completion `status.py` prints `EXPERIMENT_2_1_COMPLETE_VERIFIED` only when the GPU main job, batch and steps finished COMPLETED/0:0 and report hashes pass. It then displays the final Markdown report. If the job is still pending/running, a nonzero gate means to wait; do not resubmit it.

## If submission or a run fails

If `submit.py` fails at `sbatch --test-only`, no GPU job has been submitted and no `submission.json` exists. Capture the exact Slurm error. After a GPU job is submitted, `submission.json` records its ID; **do not resubmit training** if `status.py` cannot yet certify it. Inspect `sacct` and the logs first.

## GitHub

Code is in the local Git repository. HPC `runs/`, `outputs/`, data and weights are excluded. Commit source changes before rebuilding a release; do not upload a different release over a running experiment. GitHub publishing is separate from Slurm submission.
