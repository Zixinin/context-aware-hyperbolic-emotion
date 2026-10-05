"""Convert official WoS XLSX, run frozen preprocessors, audit and fix smoke IDs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

COMMIT = '4ea3dfb39dcee82654f443811471dd29be32e87e'
ARCHIVE_SHA = 'b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def rows_from_xlsx(content):
    with zipfile.ZipFile(io.BytesIO(content)) as book:
        strings = []
        with book.open('xl/sharedStrings.xml') as f:
            for _, node in ET.iterparse(f, events=('end',)):
                if node.tag == NS + 'si':
                    strings.append(''.join(t.text or '' for t in node.iter(NS + 't')))
                    node.clear()
        with book.open('xl/worksheets/sheet1.xml') as f:
            for _, node in ET.iterparse(f, events=('end',)):
                if node.tag != NS + 'row':
                    continue
                values = [''] * 7
                for cell in node.findall(NS + 'c'):
                    column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                    assert column in 'ABCDEFG' and len(column) == 1
                    assert cell.find(NS + 'f') is None, 'Unexpected formula'
                    value = cell.findtext(NS + 'v', '')
                    kind = cell.get('t')
                    if kind == 's':
                        value = strings[int(value)]
                    elif kind == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter(NS + 't'))
                    else:
                        assert kind in (None, 'n'), f'Unexpected cell type: {kind}'
                    values[ord(column) - ord('A')] = value
                yield int(node.attrib['r']), values
                node.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = args.archive or root / 'downloads/wos-v2/WebOfScience.zip'
    out = args.output or root / 'assets/wos-v2'
    out = out.resolve()
    upstream = root / 'original_lgsa'
    assert not out.is_relative_to(upstream.resolve())
    def git(*parts):
        return subprocess.check_output(['git', '-C', str(upstream), *parts], text=True).strip()
    assert git('rev-parse', 'HEAD') == COMMIT
    assert not git('status', '--porcelain'), 'Frozen upstream is not clean'
    assert sha(archive) == ARCHIVE_SHA
    manifest = out / 'manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        assert record['archive_sha256'] == ARCHIVE_SHA and record['upstream_commit'] == COMMIT
        for relative, checksum in record['files_sha256'].items():
            assert sha(out / relative) == checksum, relative
        print('WOS_ASSETS_VERIFIED_REUSE', flush=True)
        print(json.dumps(record['summary'], indent=2))
        return
    full = out / 'full/WebOfScience'
    smoke = out / 'smoke/WebOfScience'
    (full / 'Meta-data').mkdir(parents=True, exist_ok=True)
    smoke.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HOME=str(root / '.cache/huggingface'))
    with zipfile.ZipFile(archive) as z:
        xlsx = z.read('Meta-data/Data.xlsx')
    (full / 'Meta-data/Data.xlsx').write_bytes(xlsx)
    changed = [0] * 7
    rows = 0
    with (full / 'Meta-data/Data.txt').open('w', encoding='utf-8', newline='') as f:
        for index, values in rows_from_xlsx(xlsx):
            rows += 1
            assert index == rows, 'Nonconsecutive source Excel rows'
            if index == 1:
                assert values[:3] == ['Y1', 'Y2', 'Y'] and values[6] == 'Abstract', values
            else:
                assert all(values[k].strip() for k in [0, 1, 2, 3, 4, 6])
            clean = [v.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ') for v in values]
            changed = [n + int(a != b) for n, a, b in zip(changed, values, clean)]
            f.write('\t'.join(clean) + '\n')
    assert rows == 46986, rows
    print('XLSX_CONVERTED', rows - 1, 'whitespace_changed_cells', changed, flush=True)
    source_hashes = {}
    for filename in ['preprocess_wos.py', 'data_wos.py']:
        source = upstream / 'data/WebOfScience' / filename
        source_hashes[filename] = sha(source)
        log = out / (filename + '.log')
        with log.open('w') as f:
            result = subprocess.run([sys.executable, str(source)], cwd=full,
                                    stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:], flush=True)
            raise RuntimeError(f'{filename} failed; see {log}')
        assert sha(source) == source_hashes[filename]
        print('UPSTREAM_PREPROCESS_PASS', filename, flush=True)
    # Upstream writes taxnomy; its model reads taxonomy. Data alias only.
    shutil.copyfile(full / 'wos.taxnomy', full / 'wos.taxonomy')
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    labels = torch.load(full / 'value_dict.pt', map_location='cpu', weights_only=False)
    slots = torch.load(full / 'slot.pt', map_location='cpu', weights_only=False)
    assert set(labels) == set(range(len(labels)))
    parents = {}
    for parent, children in slots.items():
        assert parent in labels
        for child in children:
            assert child in labels and child != parent and child not in parents
            parents[child] = parent
    tops = set(labels) - set(parents)
    assert len(tops) == 7 and set(slots) == tops
    assert len(labels) == 141 and len(parents) == 134, (len(labels), len(parents))
    inverse = {v: k for k, v in labels.items()}
    taxonomy_edges = set()
    for line in (full / 'wos.taxonomy').read_text().splitlines():
        parent, *children = line.split('\t')
        if parent == 'Root':
            assert {inverse[c] for c in children} == tops
        else:
            taxonomy_edges.update((inverse[parent], inverse[c]) for c in children)
    assert taxonomy_edges == {(p, c) for c, p in parents.items()}
    total = [json.loads(line) for line in (full / 'wos_total.json').read_text().splitlines()]
    assert len(total) == 46985
    ids = np.arange(len(total))
    np.random.RandomState(7).shuffle(ids)
    train, test = train_test_split(ids, test_size=0.2, random_state=0)
    train, dev = train_test_split(train, test_size=0.2, random_state=0)
    all_ids = [int(i) for part in [train, dev, test] for i in part]
    assert len(all_ids) == len(set(all_ids)) == 46985
    split_sizes, selected = {}, {}
    for split, row_ids, expected in [('train', train, 30070), ('dev', dev, 7518), ('test', test, 9397)]:
        path = full / f'WebOfScience_{split}.json'
        lines = path.read_text().splitlines()
        assert len(lines) == len(row_ids) == expected
        for line, idx in zip(lines, row_ids):
            row = json.loads(line)
            source = total[int(idx)]
            assert row == {'token': source['doc_token'], 'label': [inverse[v] for v in source['doc_label']]}
            assert len(row['label']) == 2 and parents[row['label'][1]] == row['label'][0]
        split_sizes[split] = len(lines)
        (full / f'{split}_excel_rows.json').write_text(json.dumps([int(i) + 2 for i in row_ids]) + '\n')
        if split != 'test':
            count = 256 if split == 'train' else 128
            indices = sorted(random.Random(42).sample(range(len(lines)), count))
            (smoke / path.name).write_text('\n'.join(lines[i] for i in indices) + '\n')
            selected[split] = {'split_indices': indices, 'excel_rows': [int(row_ids[i]) + 2 for i in indices]}
    for filename in ['slot.pt', 'value_dict.pt', 'wos.taxnomy', 'wos.taxonomy']:
        shutil.copyfile(full / filename, smoke / filename)
    (smoke / 'sample_ids.json').write_text(json.dumps(selected, indent=2) + '\n')
    assert not git('status', '--porcelain')
    record = {'archive_sha256': ARCHIVE_SHA, 'upstream_commit': COMMIT,
              'preprocessor_sha256': source_hashes, 'whitespace_changed_cells': changed,
              'taxonomy_alias': 'byte-identical wos.taxnomy -> wos.taxonomy; data files only',
              'summary': {'full_split_sizes': split_sizes, 'labels': len(labels),
                          'top_labels': len(tops), 'leaf_labels': len(parents),
                          'smoke_train': 256, 'smoke_dev': 128, 'sample_seed': 42,
                          'test_used_for_smoke': False, 'gpu_smoke_passed': False},
              'files_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                               if p.is_file() and p not in (manifest, manifest.with_suffix('.json.tmp'))}}
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(manifest)
    print(json.dumps(record['summary'], indent=2), flush=True)
    print('WOS_PREPROCESS_AND_SMOKE_ASSETS_READY', flush=True)


if __name__ == '__main__':
    main()
