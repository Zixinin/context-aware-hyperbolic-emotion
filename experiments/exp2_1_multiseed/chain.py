"""One allocation, one infrastructure smoke stage, then BERT followed by LGSA."""
import os
import subprocess
import sys
from mslib import *
from evidence_ms import verify_legacy


def launch(model, seed, phase, out, log, extra=()):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    # Each subprocess uses its own dedicated interpreter; models/states never cross.
    env['PATH'] = str(PYTHONS[model].parent) + os.pathsep + env['PATH']
    env['CONDA_PREFIX'] = str(PYTHONS[model].parent.parent)
    command = [str(PYTHONS[model]), '-u', str(ROOT / 'run_model.py'), '--model', model,
               '--seed', str(seed), '--phase', phase, '--out', str(out), *map(str, extra)]
    print('START', model, seed, phase, 'log=' + str(log), flush=True)
    with log.open('x') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        print(log.read_text(errors='replace')[-12000:], flush=True)
        raise RuntimeError(f'{model} seed{seed} {phase} failed ({result.returncode}); later runs stopped; see {log}')
    check_logs([log])
    print('PASS', model, seed, phase, flush=True)


def main():
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdecimal(), 'Submit via sbatch')
    audit = load_audit()
    root = ROOT / 'runs' / ('job' + job)
    root.mkdir(parents=True, exist_ok=False)
    ledger = {'status': 'RUNNING', 'job_id': job, 'start_time': now(), 'release_sha256': release(),
              'audit_sha256': sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'), 'runs': {}, 'smokes': {}}
    path = root / 'ledger.json'
    write(path, ledger)
    try:
        # Exactly one infrastructure smoke stage, covering both changed runner paths.
        for model in ['bert', 'lgsa']:
            out, log = root / (model + '_infrastructure_smoke'), root / (model + '_smoke.log')
            launch(model, 123, 'smoke', out, log)
            ledger['smokes'][model] = verify_legacy(model, out, 123, 'smoke')
            ledger['smokes'][model]['log'] = str(log)
            write(path, ledger)
        gate = root / 'INFRASTRUCTURE_SMOKE_PASS.json'
        write(gate, {'status': 'INFRASTRUCTURE_SMOKE_PASS', 'job_id': job, 'models': ledger['smokes'],
                     'release_sha256': ledger['release_sha256'], 'audit_sha256': ledger['audit_sha256']})
        print('INFRASTRUCTURE_SMOKE_PASS: no smoke metrics enter scientific comparison', flush=True)
        for model in ['bert', 'lgsa']:
            for seed in SEEDS:
                key = model + '_seed' + str(seed)
                old = audit['models'][model]
                if seed == 42 and old['decision'] == 'REUSED_VERIFIED':
                    require(sha(OLD[model] / 'SUCCESS.json') == old['run']['success_sha256'], 'Original seed42 changed after audit')
                    ledger['runs'][key] = {'model': model, 'seed': seed, 'decision': 'REUSED_VERIFIED',
                        'reason': old['reason'], 'run_directory': str(OLD[model]), 'job_id': old['run']['job_id'],
                        'success_sha256': old['run']['success_sha256']}
                    write(path, ledger)
                    print('REUSED_VERIFIED', key, flush=True)
                    continue
                out, log = root / key, root / (key + '.log')
                extra = ['--smoke-gate', str(gate)]
                if model == 'lgsa':
                    reference = root / (key + '_raw_reference')
                    launch(model, seed, 'init', reference, root / (key + '_initialization.log'))
                    extra += ['--reference', str(reference)]
                launch(model, seed, 'full', out, log, extra)
                checked = verify_legacy(model, out, seed)
                ledger['runs'][key] = {'model': model, 'seed': seed, 'decision': 'RERUN' if seed == 42 else 'NEW_RUN',
                    'reason': old['reason'] if seed == 42 else 'Required paired seed under frozen protocol',
                    'run_directory': str(out), 'job_id': job, 'success_sha256': checked['success_sha256'],
                    'log': str(log), 'run_pass_sha256': sha(out / 'RUN_PASS.json')}
                write(path, ledger)
        require(len(ledger['runs']) == 6, 'Scientific run set incomplete')
        ledger.update(status='TRAINING_PHASES_PASS_AWAITING_SLURM_COMPLETION', end_time=now())
        write(path, ledger)
        print('TRAINING_PHASES_PASS; GPU report stage follows, then status.py verifies final Slurm COMPLETED/0:0', flush=True)
    except BaseException as exc:
        ledger.update(status='FAILED', end_time=now(), error=repr(exc))
        write(path, ledger)
        raise


if __name__ == '__main__':
    main()
