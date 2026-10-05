"""Reuse the frozen smoke run() byte-for-byte; verify only the parent-map fix."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import run_frozen_smoke as core
import prepare_parentmap_patch as patch

BASE = core.ROOT / 'outputs/lgsa_official_smoke/frozen/2984968'


def geometry_acceptance(out):
    import torch
    report = json.loads((out / 'hgs_diagnostics.json').read_text())
    vectors = torch.load(out / 'initial_geometry.pt', map_location='cpu', weights_only=False)
    frozen = torch.load(BASE / 'initial_geometry.pt', map_location='cpu', weights_only=False)
    assert torch.equal(vectors['initial'], frozen['initial']), 'Raw initial label vectors changed'
    assert vectors['parent_child_edges'] == frozen['parent_child_edges']
    assert report['map_matches'] is True and report['fine_vectors_changed'] is True
    assert report['hgs']['top_max_abs_cosine'] <= 1e-4
    families = report['hgs']['sibling_residuals']
    assert all(v['max_abs_cosine'] <= 1e-4 and v['rank'] == v['children'] for v in families.values())
    initial, hgs = vectors['initial'].double(), vectors['hgs'].double()
    torch.testing.assert_close(hgs.norm(dim=1), initial.norm(dim=1), rtol=1e-5, atol=1e-6)
    errors = {}
    for parent in families:
        p = int(parent)
        children = [int(c) for c, par in report['expected_parent_map'].items() if par == p]
        direction = hgs[p] / hgs[p].norm()
        old_projection = initial[children] @ direction
        new_projection = hgs[children] @ direction
        old_residual = initial[children] - old_projection[:, None] * direction
        new_residual = hgs[children] - new_projection[:, None] * direction
        torch.testing.assert_close(new_projection, old_projection, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new_residual.norm(dim=1), old_residual.norm(dim=1), rtol=1e-5, atol=1e-6)
        errors[parent] = float((new_projection - old_projection).abs().max())
    result = {'status': 'PATCHED_HGS_GEOMETRY_PASS', 'cosine_tolerance': 1e-4,
              'preservation_rtol': 1e-5, 'preservation_atol': 1e-6,
              'top_max_abs_cosine': report['hgs']['top_max_abs_cosine'],
              'sibling_max_abs_cosine': max(v['max_abs_cosine'] for v in families.values()),
              'projection_max_errors': errors, 'initial_vectors_equal_frozen': True,
              'scope': 'Pre-GAT HGS only; post-GAT orthogonality is not required.'}
    core.write(out / 'geometry_acceptance.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    baseline = json.loads((BASE / 'SUCCESS.json').read_text())
    assert baseline['status'] == 'FROZEN_WOS_SMOKE_PASS'
    assert baseline['harness_sha256'] == core.sha(core.ROOT / 'scripts/run_frozen_smoke.py'), 'Do not modify the frozen harness'
    source, frozen_hashes, data_root, model = core.preflight()
    assert model == baseline['model']
    assert core.sha(data_root / 'manifest.json') == baseline['dataset_manifest_sha256']
    assert frozen_hashes == baseline['source_sha256']
    working = core.ROOT / 'working_lgsa'
    working_hashes = patch.verify_working(core.ROOT)
    commit = patch.git(working, 'rev-parse', 'HEAD')
    if args.preflight:
        print('PATCHED_SMOKE_PREFLIGHT_PASS', commit)
        return
    job = os.environ.get('SLURM_JOB_ID')
    assert job and platform.system() == 'Linux', 'Submit through Slurm'
    out = core.ROOT / 'outputs/lgsa_official_smoke/patched_parentmap' / job
    out.mkdir(parents=True, exist_ok=False)
    keys = ['seed', 'batch_size', 'epochs', 'learning_rate', 'max_length', 'precision',
            'gof_loss_weight', 'gof_angle', 'gof_decay', 'threshold_protocol', 'execution_notes',
            'model', 'dataset_manifest_sha256', 'hierarchy_sha256', 'harness_sha256']
    metadata = {k: baseline[k] for k in keys}
    metadata.update(experiment_name='patched_parentmap_WoS_engineering_smoke', status='RUNNING',
        original_LGSA_commit=core.COMMIT, adaptation_commit=commit, slurm_job_id=job,
        frozen_reference_job_id='2984968', source_sha256=working_hashes,
        wrapper_sha256=core.sha(__file__), hostname=platform.node(), python=platform.python_version(),
        start_time=datetime.now(timezone.utc).isoformat())
    core.write(out / 'experiment_metadata.json', metadata)
    original_verify = core.source_state
    # Override only the source-integrity checker, never a model or training function.
    def verify(path):
        if Path(path).resolve() == source.resolve():
            return original_verify(path)
        assert Path(path).resolve() == working.resolve()
        return patch.verify_working(core.ROOT)
    core.source_state = verify
    try:
        with (out / 'requirements-installed.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, stdout=f)
        assert (out / 'requirements-installed.txt').read_text() == (BASE / 'requirements-installed.txt').read_text(), 'Environment changed'
        result = core.run(out, (working, working_hashes, data_root, model))
        accepted = geometry_acceptance(out)
        metadata.update(result)
        metadata.update(status='PATCHED_WOS_SMOKE_PASS', hgs_geometry_passed=True,
                        geometry_scope=accepted['scope'])
        assert core.sha(core.ROOT / 'scripts/run_frozen_smoke.py') == baseline['harness_sha256']
        assert original_verify(source) == frozen_hashes
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error))
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        core.source_state = original_verify
        metadata['end_time'] = datetime.now(timezone.utc).isoformat()
        core.write(out / 'experiment_metadata.json', metadata)
    core.write(out / 'SUCCESS.json', metadata)
    print('PATCHED_HGS_GEOMETRY_PASS', flush=True)
    print('PATCHED_WOS_SMOKE_PASS', out, flush=True)


if __name__ == '__main__':
    main()
