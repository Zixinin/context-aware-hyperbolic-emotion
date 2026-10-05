"""Package committed Experiment 2.1 source only; exclude all data, weights and runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / 'experiments/exp2_1_multiseed'
EXCLUDED = {'audit', 'runs', 'outputs', '__pycache__', '.cache', 'logs'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', action='store_true', help='Local checks only; this archive cannot be submitted to HPC')
    args = p.parse_args()
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and not any(x in EXCLUDED or x.startswith('.') for x in p.relative_to(PACKAGE).parts)
                   and p.name not in {'RELEASE.json', 'submission.json', 'submission.lock'} and p.suffix != '.pyc')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if not args.development:
        for f in files:
            relative = str(f.relative_to(REPO))
            original = subprocess.run(['git', '-C', str(REPO), 'show', commit + ':' + relative], capture_output=True)
            if original.returncode or original.stdout != f.read_bytes():
                raise SystemExit('Commit the source before building an HPC release: ' + relative)
    manifest = {'format': 1, 'experiment': 'FYP Experiment 2.1', 'git_commit': None if args.development else commit,
                'development_only': args.development, 'files_sha256': {str(p.relative_to(PACKAGE)): sha(p) for p in files}}
    (PACKAGE/'RELEASE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dist = REPO/'dist'; dist.mkdir(exist_ok=True)
    archive = dist/'fyp21_hpc.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for f in files + [PACKAGE/'RELEASE.json']:
            tf.add(f, arcname='FYP-Experiment-2.1-MultiSeed/' + str(f.relative_to(PACKAGE)), recursive=False)
    checksum = sha(archive)
    (dist/'fyp21_hpc.tar.gz.sha256').write_text(checksum + '  fyp21_hpc.tar.gz\n')
    print('DEVELOPMENT_ONLY' if args.development else 'COMMITTED_HPC_RELEASE_READY')
    print('source_commit:', manifest['git_commit'])
    print('files:', len(files), 'archive_bytes:', archive.stat().st_size)
    print('archive:', archive)
    print('sha256:', checksum)


if __name__ == '__main__':
    main()
