"""Prepare all LGSA dependencies without allocating a GPU; resumable wheel cache."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions')
ENV = Path('/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo')
TORCH_SHA = 'eee38239682a01ab02d3790fe194753f7070f07e1741e67fb5c3c3059d3b97b2'
REQUIREMENTS = '''torch-geometric==2.6.1
transformers==4.44.2
huggingface-hub==0.26.5
datasets==2.21.0
numpy==1.26.4
scipy==1.14.1
scikit-learn==1.5.2
pandas==2.2.3
matplotlib==3.9.2
tqdm==4.66.5
PyYAML==6.0.2
swanlab==0.6.12
'''


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fetch_wheel(url, checksum, target):
    if not re.fullmatch(r'[0-9a-f]{64}', checksum):
        raise ValueError('Missing valid source SHA256')
    if target.exists():
        if digest(target) != checksum:
            raise ValueError(f'Existing file has wrong hash: {target}; inspect without deleting other downloads')
        print('VERIFIED_SKIP', target.name, flush=True)
        return
    part = target.with_name(target.name + '.part')
    identity = target.with_name(target.name + '.part.sha256')
    if identity.exists():
        if identity.read_text().strip() != checksum:
            raise ValueError(f'Partial file version changed: {part}; inspect before continuing')
    elif part.exists():
        raise ValueError(f'Partial file has no recorded identity: {part}')
    else:
        identity.write_text(checksum + '\n')
    if not (part.exists() and digest(part) == checksum):
        run(['curl', '--fail', '--location', '--continue-at', '-', '--retry', '10',
             '--retry-delay', '5', '--connect-timeout', '30', '--output', part, url])
    if digest(part) != checksum:
        raise ValueError(f'SHA256 mismatch: {part}; file retained, installation stopped')
    os.replace(part, target)
    identity.unlink()
    print('VERIFIED_COMPLETE', target.name, flush=True)


def package_file(name, version, metadata_dir):
    target = metadata_dir / f'{name}-{version}.json'
    if not target.exists():
        temp = target.with_suffix('.json.tmp')
        run(['curl', '--fail', '--location', '--retry', '10', '--retry-delay', '3',
             '--connect-timeout', '30', '--output', temp,
             f'https://pypi.org/pypi/{name}/{version}/json'])
        json.loads(temp.read_text())
        os.replace(temp, target)
    data = json.loads(target.read_text())
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    candidates = []
    for item in data['urls']:
        if item['packagetype'] != 'bdist_wheel' or item.get('yanked'):
            continue
        tags = parse_wheel_filename(item['filename'])[3]
        scores = [ranks[tag] for tag in tags if tag in ranks]
        if scores:
            candidates.append((min(scores), item['filename'], item))
    if not candidates:
        raise RuntimeError(f'No compatible binary wheel: {name}=={version}; stop without compiling')
    return min(candidates, key=lambda value: value[:2])[2]


def main():
    assert Path(sys.prefix).resolve() == ENV.resolve(), 'Wrong Python environment'
    assert sys.version_info[:2] == (3, 10), 'Python 3.10 required'
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PIP_CACHE_DIR'] = str(ROOT / '.cache/pip')
    os.environ['UV_CACHE_DIR'] = str(ROOT / '.cache/uv')
    os.environ['UV_PYTHON_DOWNLOADS'] = 'never'
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    info = ROOT / 'environment'
    wheels = ROOT / 'downloads/wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    metadata_dir = ROOT / 'downloads/metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (info / '.install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        upstream = ROOT / 'original_lgsa'
        head = run(['git', '-C', upstream, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        assert head == '4ea3dfb39dcee82654f443811471dd29be32e87e'
        assert not run(['git', '-C', upstream, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip(), 'Frozen upstream is not clean'
        torch_wheel = ROOT / 'downloads/torch/torch-2.5.1+cu118-cp310-cp310-linux_x86_64.whl'
        assert torch_wheel.is_file() and digest(torch_wheel) == TORCH_SHA, 'Verified Torch wheel required'
        req = info / 'requirements-terminal.txt'
        req.write_text(REQUIREMENTS + str(torch_wheel) + '\n')
        fingerprint = hashlib.sha256((REQUIREMENTS + TORCH_SHA + sys.version).encode()).hexdigest()
        report_path = info / 'dependency-versions.txt'
        fingerprint_path = info / 'dependency-plan.fingerprint'
        pip = [sys.executable, '-m', 'pip']
        # A small resolver-only tool avoids pip dry-run downloading all large
        # CUDA wheels into temporary directories before resumable downloads.
        uv_item = package_file('uv', '0.8.22', metadata_dir)
        uv_wheel = wheels / uv_item['filename']
        fetch_wheel(uv_item['url'], uv_item['digests']['sha256'], uv_wheel)
        run(pip + ['install', '--no-index', '--no-deps', uv_wheel])
        if not (report_path.exists() and fingerprint_path.exists() and fingerprint_path.read_text().strip() == fingerprint):
            temporary = info / 'dependency-versions.tmp.txt'
            print('Resolving dependencies; no installation yet.', flush=True)
            run([ENV / 'bin/uv', 'pip', 'compile', req, '--python', sys.executable,
                 '--only-binary', ':all:', '--index-url', 'https://pypi.org/simple',
                 '--output-file', temporary])
            os.replace(temporary, report_path)
            fingerprint_path.write_text(fingerprint + '\n')
        pins = [f'torch==2.5.1+cu118 --hash=sha256:{TORCH_SHA}']
        for line in report_path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            # uv can preserve the input wheel as a bare absolute path.
            if line.strip().startswith('/'):
                assert Path(line.strip()).resolve() == torch_wheel.resolve(), 'Unexpected local dependency path'
                line = f'torch @ {torch_wheel.as_uri()}'
            requirement = Requirement(line.strip())
            if requirement.name.lower() == 'torch':
                assert requirement.url and requirement.url.startswith('file:')
                assert Path(unquote(urlsplit(requirement.url).path)).resolve() == torch_wheel.resolve()
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            specs = list(requirement.specifier)
            assert len(specs) == 1 and specs[0].operator == '=='
            version = specs[0].version
            name = requirement.name
            item = package_file(name, version, metadata_dir)
            checksum = item['digests']['sha256']
            fetch_wheel(item['url'], checksum, wheels / item['filename'])
            pins.append(f'{name}=={version} --hash=sha256:{checksum}')
        resolved = info / 'requirements-resolved-hashes.txt'
        resolved.write_text('\n'.join(sorted(pins)) + '\n')
        print('All wheels verified; installing from local files.', flush=True)
        run(pip + ['install', '--no-index', '--find-links', wheels,
                   '--find-links', torch_wheel.parent, '--require-hashes', '-r', resolved])
        run(pip + ['check'])
        run([sys.executable, '-u', info / 'check_cpu.py'])


if __name__ == '__main__':
    main()
