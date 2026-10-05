"""Submit one GPU job containing training, scientific checks and report generation."""
import fcntl
import subprocess
from mslib import *


def main():
    require(sys.platform == 'linux', 'Submit on HPC')
    audit = load_audit()
    subprocess.run([sys.executable, str(ROOT / 'static_check.py')], check=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    script = ROOT / 'slurm/train_chain.sbatch'
    subprocess.run(['sbatch', '--test-only', '--chdir=' + str(ROOT), str(script)], check=True)
    with (ROOT / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (ROOT / 'submission.json').exists(), 'A submission already exists. Inspect it; do not duplicate GPU jobs.')
        result = {'status': 'SUBMITTING', 'release_sha256': release(), 'audit_sha256': sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'), 'created_at': now()}
        job = capture(['sbatch', '--parsable', '--chdir=' + str(ROOT), str(ROOT / 'slurm/train_chain.sbatch')]).split(';')[0]
        require(job.isdecimal(), 'Unexpected sbatch response')
        result.update(train_job=job, status='GPU_JOB_SUBMITTED')
        write(ROOT / 'submission.json', result)
        print('GPU_JOB=' + job)
        print('One gpu4090 allocation: smoke -> full runs -> report checks and figures')
        subprocess.run(['squeue', '-j', job, '-o', '%.18i %.20j %.12T %.12M %R'], check=True)


if __name__ == '__main__':
    main()
