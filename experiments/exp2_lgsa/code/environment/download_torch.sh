#!/bin/bash
# Terminal preparation only: no GPU allocation, package installation or training.
# Repeat this exact command after interruption; keep .part and completed files.
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
mkdir -p downloads/torch
cd downloads/torch

TORCH_FILE='torch-2.5.1+cu118-cp310-cp310-linux_x86_64.whl'
TORCH_SHA='eee38239682a01ab02d3790fe194753f7070f07e1741e67fb5c3c3059d3b97b2'
TORCH_URL='https://download-r2.pytorch.org/whl/cu118/torch-2.5.1%2Bcu118-cp310-cp310-linux_x86_64.whl'

if [ -f "$TORCH_FILE" ]; then
    printf '%s  %s\n' "$TORCH_SHA" "$TORCH_FILE" | sha256sum --check
else
    if ! { [ -f "$TORCH_FILE.part" ] && printf '%s  %s\n' "$TORCH_SHA" "$TORCH_FILE.part" | sha256sum --check --status; }; then
        curl --fail --location --continue-at - \
            --retry 10 --retry-delay 5 --connect-timeout 30 \
            --output "$TORCH_FILE.part" "$TORCH_URL"
    fi
    printf '%s  %s\n' "$TORCH_SHA" "$TORCH_FILE.part" | sha256sum --check
    mv "$TORCH_FILE.part" "$TORCH_FILE"
fi
printf '%s  %s\n' "$TORCH_SHA" "$TORCH_FILE" > "$TORCH_FILE.sha256"
echo 'TORCH_WHEEL_DOWNLOAD_VERIFIED'
