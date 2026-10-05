#!/bin/bash -l
#SBATCH --job-name=lgsa_create_env
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
nvidia-smi
LGSA_ENV=/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo
if [[ -e "$LGSA_ENV" ]]; then
    echo "Environment path already exists; stop for inspection." >&2
    exit 1
fi
conda create -p "$LGSA_ENV" python=3.10 pip -y --override-channels -c defaults
touch "$LGSA_ENV/.lgsa-owned"
set +u
conda activate "$LGSA_ENV"
set -u
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
[[ "$CONDA_PREFIX" == "$LGSA_ENV" ]]
PYTHON=/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python
"$PYTHON" --version
"$PYTHON" -m pip --version
"$PYTHON" -m pip check
conda list -p "$LGSA_ENV" > "environment/create-env-${SLURM_JOB_ID}.txt"
echo "LGSA_MINIMAL_ENV_PASS"
date --iso-8601=seconds
