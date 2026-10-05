"""Prepare a pinned, verified BERT-uncased snapshot; do not load/train a model."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

ROOT = Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions')
REPO = 'google-bert/bert-base-uncased'
FILES = ['config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.txt']


def main():
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    cache = ROOT / '.cache/huggingface/hub'
    pin = assets / 'smoke_bert_uncased_revision.json'
    if pin.exists():
        record = json.loads(pin.read_text())
        assert record['repo_id'] == REPO
    else:
        revision = HfApi().model_info(REPO).sha
        record = {'repo_id': REPO, 'revision': revision}
        assert re.fullmatch(r'[0-9a-f]{40}', revision)
        temporary = pin.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(pin)
    assert re.fullmatch(r'[0-9a-f]{40}', record['revision'])
    print('PINNED_MODEL', record, flush=True)
    for attempt in range(1, 6):
        try:
            snapshot = Path(snapshot_download(
                REPO, revision=record['revision'], cache_dir=str(cache),
                allow_patterns=FILES, max_workers=1, force_download=False))
            break
        except Exception as error:
            print(f'Download attempt {attempt}/5: {error}', flush=True)
            if attempt == 5:
                raise
            time.sleep(10)
    hashes = {}
    for name in FILES:
        path = snapshot / name
        assert path.is_file(), f'Missing model asset: {name}'
        expected = path.resolve().name
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha256.update(block)
                git_sha1.update(block)
        actual = sha256.hexdigest() if len(expected) == 64 else git_sha1.hexdigest()
        assert re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', expected), 'Unexpected Hub blob name'
        assert actual == expected, f'Hash mismatch: {name}; file retained for inspection'
        hashes[name] = sha256.hexdigest()
        print('HASH_OK', name, flush=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    assert config.model_type == 'bert' and config.hidden_size == 768
    assert tokenizer.vocab_size == config.vocab_size == 30522
    assert tokenizer.do_lower_case is True
    assert tokenizer.mask_token_id is not None
    record.update(model_snapshot=str(snapshot), model_sha256=hashes,
                  purpose='frozen_and_patched_WoS_engineering_smoke',
                  backbone_basis='HPT train.py default: bert-base-uncased',
                  gpu_model_load_test_passed=False)
    manifest = assets / 'smoke_bert_uncased_manifest.json'
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(manifest)
    print('MODEL_PATH', snapshot, flush=True)
    print('WOS_SMOKE_MODEL_ASSETS_READY', flush=True)


if __name__ == '__main__':
    main()
