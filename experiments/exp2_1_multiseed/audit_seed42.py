"""Step 1 BERT audit, then step 2 LGSA audit, with no training or original-file writes."""
import argparse
import os
import subprocess
import sys
from mslib import *


def inspect(model, destination):
    from evidence_ms import asset_data, verify_legacy
    current_sources(model)
    environment = expected_environment(model)
    manifest, data = asset_data(model)
    result = {'model': model, 'seed': 42, 'current_protocol_ready': True,
              'environment': environment, 'dataset_signatures': {s: data[s][0] for s in SIZES},
              'raw_manifest': manifest, 'audited_at': now()}
    if model == 'lgsa':
        subprocess.run([sys.executable, str(PROJECTS['lgsa'] / 'goemotions_adaptation/upstream_evidence.py')], check=True)
        result['patched_wos_accounting'] = sacct('2985243')
    # A missing scheduler record is an audit blocker, not a reason to wastefully retrain.
    job = '2968116' if model == 'bert' else '2985311'
    accounting = sacct(job)
    stem = 'slurm-bert42-' if model == 'bert' else 'lgsa_goemo_s42-'
    logs = check_logs([PROJECTS[model] / 'logs' / (stem + job + ext) for ext in ['.out', '.err']])
    result.update(accounting=accounting, logs_sha256=logs)
    try:
        result['run'] = verify_legacy(model, OLD[model], data=data)
        if model == 'lgsa':
            old = read(OLD[model] / 'SUCCESS.json')
            smoke_path = PROJECTS['lgsa'] / 'outputs/lgsa_goemotions/smoke/2985311'
            smoke = read(smoke_path / 'SUCCESS.json')
            verify_legacy(model, smoke_path, phase='smoke')
            require(all(old[k] == smoke[k] for k in ['initial_backbone_sha256', 'initial_model_sha256']), 'Old full did not restart from original initialization')
            result['raw_initialization_evidence'] = 'Original full/smoke initial backbone AND full model hashes equal; raw model hashes verified'
        else:
            # Historical run lacks an initial tensor digest, but its exact source and raw
            # manifest are available and verified. Do not invent a historical tensor hash.
            result['raw_initialization_evidence'] = 'Verified full source SHA + raw asset manifest + independent raw from_pretrained call; historical tensor digest was not recorded'
        result.update(decision='REUSED_VERIFIED', reason='Frozen protocol, original source, artifacts, predictions, model/data and completed Slurm evidence verified')
    except (RuntimeError, KeyError, FileNotFoundError, ValueError) as exc:
        result.update(decision='RERUN_REQUIRED', reason=str(exc))
    write(destination, result)
    print(model.upper(), result['decision'], result['reason'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inspect', choices=['bert', 'lgsa'])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    release_sha = release()
    require(read(ROOT / 'RELEASE.json').get('development_only') is False, 'Build a committed production release before the HPC audit')
    if args.inspect:
        inspect(args.inspect, args.output)
        return
    require(sys.platform == 'linux', 'Run remote audit on HPC using existing environments')
    target = ROOT / 'audit' / ('audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    target.mkdir(parents=True, exist_ok=False)
    results = {}
    for model in ['bert', 'lgsa']:
        path = target / (model + '.json')
        print('AUDITING', model.upper(), flush=True)
        subprocess.run([str(PYTHONS[model]), '-u', str(ROOT / 'audit_seed42.py'), '--inspect', model, '--output', str(path)], check=True)
        results[model] = read(path)
    require(results['bert']['dataset_signatures'] == results['lgsa']['dataset_signatures'], 'BERT/LGSA data text, IDs, labels or split order differ')
    result = {'status': 'SEED42_AUDIT_PASS', 'release_sha256': release_sha, 'created_at': now(), 'models': results,
              'execution_order': ['one infrastructure smoke stage', 'BERT missing seeds', 'LGSA missing seeds',
                                  'wait for COMPLETED/0:0', 'verify six runs', 'summarize', 'STOP']}
    write(target / 'SEED42_AUDIT_PASS.json', result)
    write(ROOT / 'audit/SEED42_AUDIT_PASS.json', result)
    print('SEED42_AUDIT_PASS', flush=True)
    for model in results:
        seeds = [123, 2026] if results[model]['decision'] == 'REUSED_VERIFIED' else [42, 123, 2026]
        print('PLAN', model.upper(), 'train seeds', seeds, '; seed42', results[model]['decision'], flush=True)


if __name__ == '__main__':
    main()
