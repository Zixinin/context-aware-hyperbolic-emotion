"""User-operated Slurm submission; handle expired afterok IDs by strict evidence checks."""
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, PACKAGE, UPSTREAM_JOB, require, write, now, source_record
from gates import upstream_gate


def main():
    subprocess.run([sys.executable, str(PACKAGE / 'preflight.py')], check=True)
    existing = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-n', 'lgsa_goemo_s42', '-o', '%i %T'], text=True).strip()
    require(not existing, 'A GoEmotions chain job already exists: ' + existing)
    record = ROOT / 'logs/goemotions-chain-submission.json'
    require(not record.exists(), 'Submission already recorded; inspect existing job before resubmitting')
    # A known failed dependency must never be converted to an unconditional job.
    lines = subprocess.check_output(['sacct', '-n', '-P', '-j', UPSTREAM_JOB,
                                     '--format=JobIDRaw,State,ExitCode'], text=True).splitlines()
    states = {parts[0]: (parts[1].strip(), parts[2].strip()) for line in lines if len(parts := line.split('|')) >= 3}
    state = states.get(UPSTREAM_JOB)
    require(state is not None, 'No accounting record for upstream job; inspect before submitting')
    active = state[0] in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'}
    complete = state == ('COMPLETED', '0:0')
    require(active or complete, 'Upstream did not succeed/is not active: ' + str(state))
    if complete:
        require(states.get(UPSTREAM_JOB + '.batch') == ('COMPLETED', '0:0'), 'Upstream batch step did not pass')
        upstream_gate()
    # Try the requested dependency while Slurm retains its job record.
    command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--dependency=afterok:' + UPSTREAM_JOB,
               str(PACKAGE / 'lgsa_goemotions_chain.sbatch')]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    mode = 'afterok:' + UPSTREAM_JOB
    if result.returncode != 0:
        if complete and ('Invalid job id' in result.stderr or 'Job dependency problem' in result.stderr):
            # Slurm MinJobAge may have expired during adapter preparation. Success is rechecked,
            # and remains mandatory at runtime; no afterany/failure bypass is permitted.
            upstream_gate()
            mode = 'completed_upstream_verified_before_submission_and_at_runtime'
            print('UPSTREAM_ALREADY_COMPLETED_AND_VERIFIED; dependency record expired', flush=True)
            result = subprocess.run(['sbatch', '--parsable', str(PACKAGE / 'lgsa_goemotions_chain.sbatch')],
                                    cwd=ROOT, text=True, capture_output=True)
        require(result.returncode == 0, 'Submission failed: ' + result.stderr)
    job = result.stdout.strip().split(';')[0]
    require(job.isdecimal(), 'Uncertain sbatch response; inspect squeue before retry: ' + result.stdout)
    write(record, {'job_id': job, 'upstream_job_id': UPSTREAM_JOB, 'mode': mode, 'submitted_at': now(),
                   'adapter_source': source_record()})
    print('SUBMITTED_GOEMOTIONS_CHAIN_JOB=' + job, flush=True)
    subprocess.run(['squeue', '-j', job, '-o', '%.18i %.22j %.12T %.12M %R'], check=True)


if __name__ == '__main__':
    main()
