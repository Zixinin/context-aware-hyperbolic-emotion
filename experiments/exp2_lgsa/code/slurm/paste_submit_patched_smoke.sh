#!/bin/bash
# Paste the whole block into the HPC terminal.
(
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
mkdir -p scripts slurm logs
if [ -n "$(squeue -h -u "$USER" -n lgsa_wos_parentmap)" ]; then
    echo "An lgsa_wos_parentmap job is already queued/running; nothing submitted."
    squeue -u "$USER" -n lgsa_wos_parentmap
    exit 1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cat > scripts/prepare_parentmap_patch.py <<'PATCH_PREP_PY'
"""Create and verify an isolated, committed, one-line LGSA model patch."""
import argparse
import json
from pathlib import Path
import subprocess

import run_frozen_smoke as core

BRANCH = 'lgsa-goemotions-parentmap'
OLD = b'value2slot = {child: parent for child, parent in self.path_list'
NEW = b'value2slot = {child: parent for parent, child in self.path_list'
DOC = '''# Minimal parent-map direction fix

Base: 4ea3dfb39dcee82654f443811471dd29be32e87e.
Frozen WoS engineering smoke 2984968 passed, but map_matches=false and
fine_vectors_changed=false. train.py supplies (parent, child) edges.

Only model change: models/prompt.py, Prompt.init_embedding:
```diff
-        value2slot = {child: parent for child, parent in self.path_list
+        value2slot = {child: parent for parent, child in self.path_list
```

Filters, model parameters, freezing, GAT, losses and thresholds are unchanged.
original_lgsa remains untouched. This branch starts from raw pretrained
bert-base-uncased for the same 256/128 WoS engineering smoke, seed 42,
batch 4, 1 epoch, same frozen run() function as job 2984968.
No Experiment 1 or WoS fine-tuned checkpoint is used for initialization.
No GoEmotions adapter has been implemented.

Acceptance: exact map, changed fine vectors, full residual rank, pre-GAT
top/sibling residual absolute cosine <= 1e-4, norm/projection/residual-norm
preservation, engineering training and checkpoint reload. GPU result pending.
'''


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def verify_working(root):
    frozen, work = root / 'original_lgsa', root / 'working_lgsa'
    original = core.source_state(frozen)
    assert work.resolve() != frozen.resolve()
    assert git(work, 'branch', '--show-current') == BRANCH
    assert not git(work, 'status', '--porcelain'), 'Working branch must be committed and clean'
    assert git(work, 'rev-parse', 'HEAD^') == core.COMMIT, 'Expected exactly one patch commit'
    names = subprocess.check_output(['git', '-C', str(work), 'ls-files', '-z']).decode().split('\0')
    files = {name: core.sha(work / name) for name in names if name}
    assert set(files) == set(original) | {'docs/code_changes.md'}
    before = (frozen / 'models/prompt.py').read_bytes()
    assert before.count(OLD) == 1
    assert (work / 'models/prompt.py').read_bytes() == before.replace(OLD, NEW, 1)
    assert (work / 'docs/code_changes.md').read_text() == DOC
    for name, digest in original.items():
        if name != 'models/prompt.py':
            assert files[name] == digest, f'Unexpected code change: {name}'
    return files


def prepare(root):
    frozen, work = root / 'original_lgsa', root / 'working_lgsa'
    original = core.source_state(frozen)
    if not work.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(frozen), str(work)], check=True)
        subprocess.run(['git', '-C', str(work), 'checkout', '-b', BRANCH, core.COMMIT], check=True)
    assert work.resolve() != frozen.resolve()
    assert git(work, 'branch', '--show-current') == BRANCH
    if git(work, 'rev-parse', 'HEAD') == core.COMMIT:
        before = (frozen / 'models/prompt.py').read_bytes()
        assert before.count(OLD) == 1
        after = before.replace(OLD, NEW, 1)
        target = work / 'models/prompt.py'
        assert target.read_bytes() in (before, after), 'Unexpected existing prompt.py'
        target.write_bytes(after)
        doc = work / 'docs/code_changes.md'
        doc.parent.mkdir(exist_ok=True)
        if doc.exists():
            assert doc.read_text() == DOC
        doc.write_text(DOC)
        changed = set(git(work, 'diff', '--name-only', core.COMMIT).splitlines())
        assert 'models/prompt.py' in changed and changed <= {'models/prompt.py', 'docs/code_changes.md'}, changed
        untracked = set(git(work, 'ls-files', '--others', '--exclude-standard').splitlines())
        assert untracked <= {'docs/code_changes.md'}, untracked
        subprocess.run(['git', '-C', str(work), 'add', 'models/prompt.py', 'docs/code_changes.md'], check=True)
        subprocess.run(['git', '-C', str(work), '-c', 'user.name=LGSA experiment automation',
                        '-c', 'user.email=lgsa-experiment@localhost', 'commit',
                        '-m', 'Fix parent-child unpacking in HGS initialization'], check=True)
    verify_working(root)
    assert core.source_state(frozen) == original
    patch = git(work, 'diff', core.COMMIT, 'HEAD', '--', 'models/prompt.py') + '\n'
    print(patch, flush=True)
    print('PARENTMAP_PATCH_COMMITTED', git(work, 'rev-parse', 'HEAD'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=core.ROOT)
    args = parser.parse_args()
    baseline = json.loads((args.root / 'outputs/lgsa_official_smoke/frozen/2984968/SUCCESS.json').read_text())
    assert baseline['status'] == 'FROZEN_WOS_SMOKE_PASS'
    assert baseline['checkpoint_reload_logits_match'] and baseline['optimizer_steps'] == 64
    assert baseline['harness_sha256'] == core.sha(args.root / 'scripts/run_frozen_smoke.py')
    prepare(args.root)
PATCH_PREP_PY
cat > scripts/run_patched_smoke.py <<'PATCH_RUN_PY'
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
PATCH_RUN_PY
cat > slurm/lgsa_patched_smoke.sbatch <<'PATCH_SLURM'
#!/bin/bash -l
#SBATCH --job-name=lgsa_wos_parentmap
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
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
set -u
date --iso-8601=seconds
hostname
nvidia-smi
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/run_patched_smoke.py
date --iso-8601=seconds
PATCH_SLURM
bash -n slurm/lgsa_patched_smoke.sbatch
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/prepare_parentmap_patch.py
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/run_patched_smoke.py --preflight
job_reply=$(sbatch --parsable slurm/lgsa_patched_smoke.sbatch)
job_id=${job_reply%%;*}
printf '%s\n' "$job_id" > logs/patched-smoke-last-jobid.txt
printf 'SUBMITTED_PATCHED_SMOKE_JOB=%s\n' "$job_id"
squeue -j "$job_id" -o "%.18i %.22j %.12T %.12M %R"
)
