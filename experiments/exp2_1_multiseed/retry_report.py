"""Recover only a missing/failed CPU report job; never submits GPU training."""
import fcntl
from mslib import *


def main():
    load_audit()
    with (ROOT / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = read(ROOT / 'submission.json')
        require('train_job' in s, 'No training submission to report')
        # Reporting can be recovered after actual successful training, not by
        # changing dependencies around a failed allocation.
        sacct(s['train_job'])
        if 'report_job' in s:
            rows = capture(['sacct', '-S', '2026-09-01', '-j', s['report_job'], '--noheader', '--parsable2', '--format=JobIDRaw,State']).splitlines()
            match = [line.split('|')[1] for line in rows if line.split('|')[0] == s['report_job']]
            require(len(match) == 1 and match[0].split()[0] in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'},
                    'Previous report is not known failed; do not duplicate it')
            s.setdefault('previous_report_jobs', []).append(s['report_job'])
        job = capture(['sbatch', '--parsable', '--dependency=afterok:' + s['train_job'], '--kill-on-invalid-dep=yes',
                       '--export=ALL,FYP21_TRAIN_JOB=' + s['train_job'], '--chdir=' + str(ROOT), str(ROOT/'slurm/finalize.sbatch')]).split(';')[0]
        require(job.isdecimal(), 'Unexpected sbatch response')
        s.update(report_job=job, status='REPORT_RESUBMITTED', report_resubmitted_at=now())
        write(ROOT/'submission.json', s)
        print('REPORT_JOB=' + job, '; existing successful training preserved')


if __name__ == '__main__':
    main()
