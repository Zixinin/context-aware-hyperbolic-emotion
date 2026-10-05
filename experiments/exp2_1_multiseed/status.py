"""Read-only queue/accounting/result status; never equates a queued job with success."""
from mslib import *


def main():
    submission = read(ROOT / 'submission.json')
    print(json.dumps(submission, indent=2))
    jobs = [submission[k] for k in ['train_job', 'report_job'] if k in submission]
    subprocess.run(['squeue', '-j', ','.join(jobs), '-o', '%.18i %.20j %.12T %.12M %R'], check=False)
    subprocess.run(['sacct', '-S', '2026-09-01', '-j', ','.join(jobs), '--format=JobID,JobName%20,State,Elapsed,ExitCode,MaxRSS'], check=False)
    ledger = ROOT / 'runs' / ('job' + submission['train_job']) / 'ledger.json'
    if ledger.exists():
        record = read(ledger)
        print('Training ledger:', record['status'])
        for key, r in record.get('runs', {}).items():
            print(key, r['decision'], r['run_directory'])
    latest = ROOT / 'outputs/LATEST.json'
    if not latest.exists():
        print('FINAL_RESULT_NOT_READY; consult logs and accounting above')
        return
    r = read(latest)
    require(r['train_job'] == submission['train_job'], 'Stale latest-result pointer')
    sacct(submission['train_job'])
    require('report_job' in submission, 'No report submission recorded')
    sacct(submission['report_job'])
    check_logs([ROOT / 'logs' / (f'fyp21_report-{submission["report_job"]}' + ext) for ext in ['.out', '.err']])
    report = Path(r['report'])
    artifacts(report)
    print('EXPERIMENT_2_1_COMPLETE_VERIFIED')
    print('Report:', report / 'MULTISEED_SUMMARY.md')
    print((report / 'MULTISEED_SUMMARY.md').read_text())


if __name__ == '__main__':
    main()
