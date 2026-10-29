"""Explicit human-authorized deferred backup, never a fabricated backup PASS."""
import json,shutil
from pathlib import Path

POLICY={
    'authorization_date':'2026-10-29',
    'authorization':'Human requested: start this experiment now without external disk; retrieve complete new artifacts to Mac after training',
    'external_drive':'NOT_ATTEMPTED_USER_AUTHORIZED_DEFER',
    'mac_local':'PENDING_USER_TRANSFER_AND_TARGET_SHA256_VERIFICATION',
    'historical_external_backup':'UNVERIFIED_IN_CURRENT_SESSION',
    'github_upload':'NOT_PERFORMED_BY_THIS_PIPELINE',
    'new_Test':False,'new_seeds':False,
}

def check_policy(root):
    root=Path(root)
    if json.loads((root/'USER_AUTHORIZED_TRANSFER_POLICY.json').read_text(encoding='utf-8'))!=POLICY:
        raise RuntimeError('Missing/mismatching explicit deferred-transfer policy')
    if shutil.disk_usage(root).free<10*1024**3:
        raise RuntimeError('Need at least10GiB Ubuntu free space; no files deleted')
    return dict(POLICY)
