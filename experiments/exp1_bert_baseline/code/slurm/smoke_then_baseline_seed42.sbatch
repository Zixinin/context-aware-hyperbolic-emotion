#!/bin/bash -l
#SBATCH --job-name=goemo_full42
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-bert42-%j.out
#SBATCH --error=logs/slurm-bert42-%j.err

# One allocation: smoke -> verified artifacts -> fresh full baseline.
# The parent Slurm job stays RUNNING between stages; check sacct after both finish.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
source "$PROJECT_ROOT/hpc/job_start.sh"

if [[ -e outputs/bert_baseline/seed42 ]]; then
    echo "Existing outputs/bert_baseline/seed42: inspect/preserve it before resubmission." >&2
    exit 1
fi
"$PYTHON" -c 'from src.common import load_config; assert load_config()["seed"] == 42, "This launcher is for seed 42 only"'

echo "SMOKE_STAGE_START job=$SLURM_JOB_ID"
"$PYTHON" -u -m src.train --mode smoke

# Separate Python process releases all smoke GPU allocations and resets RNGs
# when run() loads the original pretrained model for the full experiment.
"$PYTHON" -u - <<'PY'
import os
import subprocess
import sys

from src.common import ROOT, load_config, sha256, source_hashes
from scripts.verify_run import verify_artifacts
from src.train import run

job_id = os.environ["SLURM_JOB_ID"]
smoke_dir = ROOT / "outputs/smoke/seed42" / f"job{job_id}"
meta, smoke = verify_artifacts(smoke_dir)
config = load_config()
config["mode"] = "full"

if meta["slurm_job_id"] != job_id or smoke["mode"] != "smoke":
    raise RuntimeError("Smoke must have completed inside this same Slurm job.")
if meta["source_sha256"] != source_hashes():
    raise RuntimeError("Code/config changed since smoke; aborting full baseline.")
if meta["assets_manifest_sha256"] != sha256(ROOT / "assets/manifest.json"):
    raise RuntimeError("Prepared data/model manifest changed since smoke.")
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
if freeze != (smoke_dir / "pip-freeze.txt").read_text():
    raise RuntimeError("Python packages changed since smoke.")
for key, value in config.items():
    if key not in ("epochs", "mode") and smoke.get(key) != value:
        raise RuntimeError(f"Configuration differs from smoke: {key}")

# Do not demand sacct COMPLETED here: the single parent allocation is still
# RUNNING. The shell has checked the smoke process exit code, and above we
# verified its SUCCESS marker, artifact checksums, epochs and checkpoint reload.
print(f"SMOKE_STAGE_PASSED job={job_id}; same-allocation artifact gate passed", flush=True)
print("FULL_BASELINE_START: fresh pretrained BERT; complete official train split; configured full epochs", flush=True)
run(config, ROOT / "outputs/bert_baseline/seed42")
print("SMOKE_AND_FULL_BASELINE_PASS", flush=True)
PY
date -Is
