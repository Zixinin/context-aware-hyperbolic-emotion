"""GPU-stage scientific checks and provisional report; Slurm completion comes later."""
import argparse
from mslib import *
from normalize_results import normalize
from report_ms import generate
from scripts.verify_multiseed_results import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-job', required=True)
    args = p.parse_args()
    audit = load_audit()
    ledger_path = ROOT / 'runs' / ('job' + args.train_job) / 'ledger.json'
    ledger = read(ledger_path)
    require(ledger['job_id'] == args.train_job and ledger['status'] == 'TRAINING_PHASES_PASS_AWAITING_SLURM_COMPLETION', 'Training ledger incomplete')
    require(ledger['release_sha256'] == release() and ledger['audit_sha256'] == sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'), 'Ledger code/audit mismatch')
    require(set(ledger['runs']) == {f'{m}_seed{s}' for m in ['bert', 'lgsa'] for s in SEEDS}, 'Missing scientific runs')
    require(os.environ.get('SLURM_JOB_ID') == args.train_job, 'Report must run inside its GPU training job')
    accounting = {args.train_job: {'job_id': args.train_job, 'state': 'AWAITING_SLURM_COMPLETION'}}
    check_logs([ROOT / 'logs' / ('fyp21_train-' + args.train_job + ext) for ext in ['.out', '.err']])
    out = ROOT / 'outputs' / ('job' + args.train_job)
    if out.exists():
        # Preserve any partial derived output for diagnosis; never modify native runs.
        backup = out.with_name(out.name + '.previous-finalize-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        require(not backup.exists(), 'Report recovery backup already exists')
        out.rename(backup)
    out.mkdir(parents=True, exist_ok=False)
    for model in ['bert', 'lgsa']:
        current_sources(model)
        for seed in SEEDS:
            entry = ledger['runs'][f'{model}_seed{seed}']
            job = entry['job_id']
            if job not in accounting:
                accounting[job] = sacct(job)
            if entry['decision'] == 'REUSED_VERIFIED':
                for path, digest in audit['models'][model]['logs_sha256'].items():
                    require(sha(path) == digest, 'Historical log changed after audit')
            else:
                check_logs([entry['log']])
            normalize(entry, audit, out / MODEL_DIR[model] / ('seed' + str(seed)))
    records, accounting = verify(out, active_job=args.train_job)
    print('SIX_RUNS_INTEGRITY_PASS', flush=True)
    report = out / 'multiseed_comparison'
    counts = generate(records, report, provisional=True)
    write(report / 'INTEGRITY_PRELIMINARY.json', {'status': 'SIX_RUNS_ARTIFACTS_PASS_AWAITING_SLURM', 'accounting': accounting,
        'release_sha256': release(), 'ledger_sha256': sha(ledger_path), **counts})
    write(report / 'PRELIMINARY.json', {'status': 'AWAITING_SLURM_COMPLETION', 'created_at': now(),
        'training_job': args.train_job,
        'completion_note': 'status.py verifies the GPU main/batch/steps after job exit',
        'artifacts_sha256': {str(p.relative_to(report)): sha(p) for p in sorted(report.rglob('*')) if p.is_file()}})
    print('GPU_REPORT_ARTIFACTS_PASS_AWAITING_SLURM_COMPLETION', report, flush=True)
    print('STOP: no further training, ablation, or Hyperbolic experiment', flush=True)


if __name__ == '__main__':
    main()
