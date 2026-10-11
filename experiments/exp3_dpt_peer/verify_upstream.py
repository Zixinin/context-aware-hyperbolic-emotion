"""Verify the vendored sampling excerpt against the immutable official source."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '43e2daf6578ed6df885329cf229c3f5b749e19e7'
BASE = f'https://raw.githubusercontent.com/ccx06/Dual-Prompt-Tuning-for-HTC/{COMMIT}/'


def check(text):
    # File ends before compose_path; the terminal newline is the only normalization.
    return text.split('\ndef compose_path')[0].rstrip() + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-file', type=Path, help='Optional saved full upstream model_utils.py for offline audit')
    args = p.parse_args()
    raw = args.source_file.read_bytes() if args.source_file else urllib.request.urlopen(
        BASE + 'src/models/model_utils.py', timeout=30).read()
    local = Path(__file__).parent / 'upstream/dpt_sampling.py'
    expected = check(raw.decode('utf-8')).encode('utf-8')
    if expected != local.read_bytes():
        raise RuntimeError('Official DPT excerpt differs; do not train')
    certificate = {'status': 'DPT_UPSTREAM_EXCERPT_PASS', 'commit': COMMIT,
        'url': BASE + 'src/models/model_utils.py', 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'excerpt_sha256': hashlib.sha256(expected).hexdigest(), 'normalization': 'trim terminal whitespace and append one newline'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(certificate, stream, indent=2)
        stream.write('\n')
    print(json.dumps(certificate, indent=2))


if __name__ == '__main__':
    main()
