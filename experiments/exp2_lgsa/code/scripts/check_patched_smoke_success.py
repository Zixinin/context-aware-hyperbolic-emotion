"""Read-only prerequisite for a future GoEmotions job; never starts training."""
import argparse
import json
import math
from pathlib import Path

import run_frozen_smoke as core
import prepare_parentmap_patch as patch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def check(root, job, commit):
    require(job.isdecimal(), 'Invalid upstream Slurm job ID')
    out = root / 'outputs/lgsa_official_smoke/patched_parentmap' / job
    success = read(out / 'SUCCESS.json')
    require(success == read(out / 'experiment_metadata.json'), 'Final metadata mismatch')
    require(success['status'] == 'PATCHED_WOS_SMOKE_PASS', 'Patched engineering smoke did not pass')
    require(str(success['slurm_job_id']) == job, 'Wrong upstream job evidence')
    require(success['adaptation_commit'] == commit, 'Unexpected tested patch commit')
    require(success['original_LGSA_commit'] == core.COMMIT, 'Unexpected frozen commit')
    require(success['hgs_geometry_passed'] is True, 'HGS acceptance missing')
    require(success['checkpoint_reload_logits_match'] is True, 'Checkpoint reload failed')
    require(success['optimizer_steps'] == 64, 'Incomplete training smoke')
    delta = success['backbone_parameter_max_change']
    require(math.isfinite(delta) and delta > 0, 'No finite backbone update')
    require(success['end_time'], 'Run has no completion time')
    diagnostic = read(out / 'hgs_diagnostics.json')
    require(diagnostic['map_matches'] is True, 'Parent map did not match')
    require(diagnostic['fine_vectors_changed'] is True, 'Fine labels were not updated')
    require(diagnostic['actual_parent_map'] == diagnostic['expected_parent_map'], 'Parent map evidence differs')
    for family in diagnostic['hgs']['sibling_residuals'].values():
        require(family['rank'] == family['children'], 'Rank-deficient HGS residuals')
        require(0 <= family['max_abs_cosine'] <= 1e-4, 'Nonorthogonal sibling residuals')
    accepted = read(out / 'geometry_acceptance.json')
    require(accepted['status'] == 'PATCHED_HGS_GEOMETRY_PASS', 'Geometry acceptance failed')
    require(accepted['initial_vectors_equal_frozen'] is True, 'Unmatched initialization')
    for field in ('top_max_abs_cosine', 'sibling_max_abs_cosine'):
        value = accepted[field]
        require(math.isfinite(value) and 0 <= value <= 1e-4, 'HGS orthogonality failed: ' + field)
    require(core.sha(out / 'checkpoint_last.pt') == success['checkpoint_sha256'], 'Checkpoint integrity failed')
    require(core.sha(root / 'scripts/run_frozen_smoke.py') == success['harness_sha256'], 'Frozen harness changed')
    require(core.sha(root / 'scripts/run_patched_smoke.py') == success['wrapper_sha256'], 'Patched harness changed')
    require(patch.git(root / 'working_lgsa', 'rev-parse', 'HEAD') == commit, 'Working source changed since smoke')
    require(patch.verify_working(root) == success['source_sha256'], 'Tested source hashes changed')
    return {'status': 'PATCHED_PREREQUISITE_PASS', 'upstream_job_id': job,
            'tested_patch_commit': commit, 'success_sha256': core.sha(out / 'SUCCESS.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=core.ROOT)
    parser.add_argument('--job', default='2985243')
    parser.add_argument('--commit', default='209f83762dae1b354ba40e68b6952482ecf96208')
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.job, args.commit), indent=2))
