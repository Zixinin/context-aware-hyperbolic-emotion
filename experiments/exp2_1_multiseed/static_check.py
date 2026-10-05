"""Syntax and frozen-protocol assertions. No models or GPU training on import."""
import ast
import subprocess
from mslib import *


def main():
    release()
    manifest = read(ROOT / 'RELEASE.json')
    for name in manifest['files_sha256']:
        p = ROOT / name
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
        elif p.suffix in ['.sh', '.sbatch']:
            subprocess.run(['bash', '-n', str(p)], check=True)
    b, l = config('bert', 42), config('lgsa', 42)
    require(b['epochs'] == l['epochs'] == 4 and b['effective_batch_size'] == l['batch_size'] == 16, 'Budget changed')
    require(b['learning_rate'] == 5e-5 and l['learning_rate'] == 1e-4, 'Learning rate changed')
    require(b['threshold_candidates'] == l['threshold_candidates'] == CANDIDATES, 'Threshold search changed')
    require(b['max_length'] == l['text_max_length'] == 50 and l['total_max_length'] == 54, 'Token budget changed')
    for model in ['bert', 'lgsa']:
        for seed in SEEDS:
            original, actual = config(model, 42), config(model, seed)
            require({k: v for k, v in original.items() if k != 'seed'} == {k: v for k, v in actual.items() if k != 'seed'}, 'Only the seed may vary')
    # The new wrappers must call the original loops and must not introduce a new
    # backward/optimizer/scheduler/loss implementation.
    tree = ast.parse((ROOT / 'run_model.py').read_text())
    banned = {'backward', 'Adam', 'AdamW', 'clip_grad_norm_', 'binary_cross_entropy_with_logits'}
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in banned for n in ast.walk(tree)), 'New training math in wrapper')
    train_shell = (ROOT / 'slurm/train_chain.sbatch').read_text()
    require('set -euo pipefail' in train_shell and '#SBATCH --gres=gpu:1' in train_shell, 'Slurm failure/resource gate missing')
    require('python -u chain.py' in train_shell and 'python -u finalize.py --train-job "$SLURM_JOB_ID"' in train_shell,
            'GPU training must gate reporting inside the same allocation')
    submit = (ROOT / 'submit.py').read_text()
    require('finalize.sbatch' not in submit and 'report_job' not in submit, 'CPU report submission remains')
    print('MULTISEED_STATIC_CHECK_PASS', len(manifest['files_sha256']), 'release files')


if __name__ == '__main__':
    main()
