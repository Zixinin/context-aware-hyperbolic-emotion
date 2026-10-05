"""Submit only after audits/static checks. One GPU job and an afterok CPU finalizer."""
import fcntl
import subprocess
from mslib import *


def main():
    require(sys.platform == 'linux', 'Submit on HPC')
    audit = load_audit()
    subprocess.run([sys.executable, str(ROOT / 'static_check.py')], check=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    for script in ['train_chain.sbatch', 'finalize.sbatch']:
        # Validate account/partition/QOS eligibility before reserving either job.
        subprocess.run(['sbatch', '--test-only', '--chdir=' + str(ROOT), str(ROOT / 'slurm' / script)], check=True)
    with (ROOT / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (ROOT / 'submission.json').exists(), 'A submission already exists. Inspect it; do not duplicate GPU jobs.')
        result = {'status': 'SUBMITTING', 'release_sha256': release(), 'audit_sha256': sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'), 'created_at': now()}
        job = capture(['sbatch', '--parsable', '--chdir=' + str(ROOT), str(ROOT / 'slurm/train_chain.sbatch')]).split(';')[0]
        require(job.isdecimal(), 'Unexpected sbatch response')
        result.update(train_job=job, status='TRAIN_SUBMITTED_REPORT_NOT_YET_SUBMITTED')
        write(ROOT / 'submission.json', result)
        # Persist the training ID before a second submission; a report submission
        # failure must never cause the expensive training job to be duplicated.
        report = capture(['sbatch', '--parsable', '--dependency=afterok:' + job, '--kill-on-invalid-dep=yes',
                          '--export=ALL,FYP21_TRAIN_JOB=' + job, '--chdir=' + str(ROOT), str(ROOT / 'slurm/finalize.sbatch')]).split(';')[0]
        require(report.isdecimal(), 'Unexpected report sbatch response')
        result.update(report_job=report, status='CHAIN_SUBMITTED')
        write(ROOT / 'submission.json', result)
        print('TRAIN_JOB=' + job)
        print('REPORT_JOB=' + report)
        print('Report dependency: afterok:' + job)
        subprocess.run(['squeue', '-j', job + ',' + report, '-o', '%.18i %.20j %.12T %.12M %R'], check=True)


if __name__ == '__main__':
    main()
