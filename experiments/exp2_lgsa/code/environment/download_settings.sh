# Source after activating the dedicated environment on HPC.
# pip>=25.2 is required for the resume retry option; bootstrap pip separately.
export PIP_CACHE_DIR=/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions/.cache/pip
export PIP_TIMEOUT=120
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=20
export HF_HOME=/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions/.cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"
