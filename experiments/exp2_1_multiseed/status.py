"""Verify the sole GPU job's final Slurm state and certify its GPU-built report."""
import subprocess
from mslib import *
from scripts.verify_multiseed_results import verify

BANNER = '**预验收：GPU 作业仍在运行。须待主作业及 batch/全部 step 均为 COMPLETED/0:0，并运行 status.py，方可视作最终结果。**\n\n'
PRELIMINARY_END = '六组记录的科研产物、checkpoint、28 类指标与配对样本已完成作业内检查；新运行的 Slurm 终态待核对。Smoke 指标未进入任何科学汇总。未运行 loss ablation 或 Hyperbolic 实验。'
FINAL_END = '全部科研训练记录已核对 Slurm COMPLETED/0:0、checkpoint、产物哈希、28 类指标与配对样本。Smoke 指标未进入任何科学汇总。输出完成后停止；未运行 loss ablation 或 Hyperbolic 实验。'
HEADING = '# Experiment 2.1 — Multi-Seed Stability Verification\n\n'


def final_summary_text(original):
    require(BANNER in original and PRELIMINARY_END in original, 'Preliminary report status unclear')
    return original.replace(BANNER, '', 1).replace(PRELIMINARY_END, FINAL_END, 1)


def verify_preliminary_report(report, preliminary):
    for name, digest in preliminary['artifacts_sha256'].items():
        path = report / name
        require(path.is_file(), 'Preliminary report artifact missing: ' + str(path))
        if name == 'MULTISEED_SUMMARY.md' and sha(path) != digest:
            final = path.read_text()
            require(final.startswith(HEADING) and FINAL_END in final and BANNER not in final,
                    'Report changed unexpectedly')
            reconstructed = final.replace(HEADING, HEADING + BANNER, 1).replace(FINAL_END, PRELIMINARY_END, 1)
            require(hashlib.sha256(reconstructed.encode()).hexdigest() == digest, 'Preliminary report changed')
        else:
            require(sha(path) == digest, 'Preliminary report artifact changed: ' + str(path))


def promote(job, out, report):
    preliminary = read(report / 'PRELIMINARY.json')
    require(preliminary['status'] == 'AWAITING_SLURM_COMPLETION'
            and preliminary['training_job'] == job, 'Wrong preliminary report')
    verify_preliminary_report(report, preliminary)
    ledger_path = ROOT / 'runs' / ('job' + job) / 'ledger.json'
    ledger = read(ledger_path)
    require(ledger['status'] == 'TRAINING_PHASES_PASS_AWAITING_SLURM_COMPLETION', 'Training ledger incomplete')
    require(read(report / 'INTEGRITY_PRELIMINARY.json')['ledger_sha256'] == sha(ledger_path), 'Ledger changed')
    check_logs([ROOT / 'logs' / ('fyp21_train-' + job + ext) for ext in ['.out', '.err']])
    # The GPU allocation already verified native checkpoints and scientific
    # metrics. Recheck normalized data and accounting before promoting status.
    for model in ['bert', 'lgsa']:
        for seed in SEEDS:
            path = out / MODEL_DIR[model] / ('seed' + str(seed))
            m = read(path / 'experiment_metadata.json')
            if m['slurm_job_id'] != job:
                continue
            require(m['status'] in {'AWAITING_SLURM_COMPLETION', 'COMPLETED'}, 'Invalid promotion state')
            m.update(status='COMPLETED', slurm_state='COMPLETED', slurm_exit_code='0:0')
            write(path / 'experiment_metadata.json', m)
            marker = read(path / 'SUCCESS.json')
            marker['status'] = 'NORMALIZED_RUN_PASS'
            marker['artifacts_sha256']['experiment_metadata.json'] = sha(path / 'experiment_metadata.json')
            write(path / 'SUCCESS.json', marker)
    records, jobs = verify(out, check_checkpoint_hashes=False)
    require(job in jobs and len(records) == 6, 'Final six-run verification failed')
    summary = report / 'MULTISEED_SUMMARY.md'
    summary_text = summary.read_text()
    if BANNER in summary_text:
        summary.write_text(final_summary_text(summary_text))
    else:
        require(FINAL_END in summary_text and PRELIMINARY_END not in summary_text, 'Final report status unclear')
    write(report / 'INTEGRITY_PASS.json', {'status': 'SIX_RUNS_INTEGRITY_PASS',
        'accounting': jobs, 'release_sha256': release(), 'ledger_sha256': sha(ledger_path),
        'gpu_preliminary_sha256': sha(report / 'PRELIMINARY.json')})
    write(report / 'SUCCESS.json', {'status': 'MULTISEED_COMPARISON_PASS', 'created_at': now(),
        'training_job': job, 'artifacts_sha256': {str(p.relative_to(report)): sha(p)
        for p in sorted(report.rglob('*')) if p.is_file() and p.name != 'SUCCESS.json'}})
    artifacts(report)
    write(ROOT / 'outputs/LATEST.json', {'train_job': job, 'outputs': str(out), 'report': str(report)})


def main():
    path = ROOT / 'submission.json'
    require(path.exists(), 'No submission.json: the prior submit.py stopped before submitting any job')
    submission = read(path)
    print(json.dumps(submission, indent=2))
    job = submission['train_job']
    subprocess.run(['squeue', '-j', job, '-o', '%.18i %.20j %.12T %.12M %R'], check=False)
    subprocess.run(['sacct', '-S', '2026-09-01', '-j', job,
                    '--format=JobID,JobName%20,State,Elapsed,ExitCode,MaxRSS'], check=False)
    ledger_path = ROOT / 'runs' / ('job' + job) / 'ledger.json'
    if ledger_path.exists():
        record = read(ledger_path)
        print('Training ledger:', record['status'])
        for key, r in record.get('runs', {}).items():
            print(key, r['decision'], r['run_directory'])
    # This gate runs only after the job and its batch/steps actually succeed.
    sacct(job)
    out = ROOT / 'outputs' / ('job' + job)
    report = out / 'multiseed_comparison'
    latest = ROOT / 'outputs/LATEST.json'
    if not latest.exists():
        require((report / 'PRELIMINARY.json').exists(), 'GPU report did not finish; inspect logs')
        promote(job, out, report)
    pointer = read(latest)
    require(pointer['train_job'] == job and pointer['report'] == str(report), 'Stale latest-result pointer')
    success = read(report / 'SUCCESS.json')
    require(success['status'] == 'MULTISEED_COMPARISON_PASS' and success['training_job'] == job, 'Report success mismatch')
    artifacts(report)
    print('EXPERIMENT_2_1_COMPLETE_VERIFIED')
    print('Report:', report / 'MULTISEED_SUMMARY.md')
    print((report / 'MULTISEED_SUMMARY.md').read_text())


if __name__ == '__main__':
    main()
