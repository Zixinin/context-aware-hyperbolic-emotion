"""Create and verify an isolated, committed, one-line LGSA model patch."""
import argparse
import json
from pathlib import Path
import subprocess

import run_frozen_smoke as core

BRANCH = 'lgsa-goemotions-parentmap'
OLD = b'value2slot = {child: parent for child, parent in self.path_list'
NEW = b'value2slot = {child: parent for parent, child in self.path_list'
DOC = '''# Minimal parent-map direction fix

Base: 4ea3dfb39dcee82654f443811471dd29be32e87e.
Frozen WoS engineering smoke 2984968 passed, but map_matches=false and
fine_vectors_changed=false. train.py supplies (parent, child) edges.

Only model change: models/prompt.py, Prompt.init_embedding:
```diff
-        value2slot = {child: parent for child, parent in self.path_list
+        value2slot = {child: parent for parent, child in self.path_list
```

Filters, model parameters, freezing, GAT, losses and thresholds are unchanged.
original_lgsa remains untouched. This branch starts from raw pretrained
bert-base-uncased for the same 256/128 WoS engineering smoke, seed 42,
batch 4, 1 epoch, same frozen run() function as job 2984968.
No Experiment 1 or WoS fine-tuned checkpoint is used for initialization.
No GoEmotions adapter has been implemented.

Acceptance: exact map, changed fine vectors, full residual rank, pre-GAT
top/sibling residual absolute cosine <= 1e-4, norm/projection/residual-norm
preservation, engineering training and checkpoint reload. GPU result pending.
'''


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def verify_working(root):
    frozen, work = root / 'original_lgsa', root / 'working_lgsa'
    original = core.source_state(frozen)
    assert work.resolve() != frozen.resolve()
    assert git(work, 'branch', '--show-current') == BRANCH
    assert not git(work, 'status', '--porcelain'), 'Working branch must be committed and clean'
    assert git(work, 'rev-parse', 'HEAD^') == core.COMMIT, 'Expected exactly one patch commit'
    names = subprocess.check_output(['git', '-C', str(work), 'ls-files', '-z']).decode().split('\0')
    files = {name: core.sha(work / name) for name in names if name}
    assert set(files) == set(original) | {'docs/code_changes.md'}
    before = (frozen / 'models/prompt.py').read_bytes()
    assert before.count(OLD) == 1
    assert (work / 'models/prompt.py').read_bytes() == before.replace(OLD, NEW, 1)
    assert (work / 'docs/code_changes.md').read_text() == DOC
    for name, digest in original.items():
        if name != 'models/prompt.py':
            assert files[name] == digest, f'Unexpected code change: {name}'
    return files


def prepare(root):
    frozen, work = root / 'original_lgsa', root / 'working_lgsa'
    original = core.source_state(frozen)
    if not work.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(frozen), str(work)], check=True)
        subprocess.run(['git', '-C', str(work), 'checkout', '-b', BRANCH, core.COMMIT], check=True)
    assert work.resolve() != frozen.resolve()
    assert git(work, 'branch', '--show-current') == BRANCH
    if git(work, 'rev-parse', 'HEAD') == core.COMMIT:
        before = (frozen / 'models/prompt.py').read_bytes()
        assert before.count(OLD) == 1
        after = before.replace(OLD, NEW, 1)
        target = work / 'models/prompt.py'
        assert target.read_bytes() in (before, after), 'Unexpected existing prompt.py'
        target.write_bytes(after)
        doc = work / 'docs/code_changes.md'
        doc.parent.mkdir(exist_ok=True)
        if doc.exists():
            assert doc.read_text() == DOC
        doc.write_text(DOC)
        changed = set(git(work, 'diff', '--name-only', core.COMMIT).splitlines())
        assert 'models/prompt.py' in changed and changed <= {'models/prompt.py', 'docs/code_changes.md'}, changed
        untracked = set(git(work, 'ls-files', '--others', '--exclude-standard').splitlines())
        assert untracked <= {'docs/code_changes.md'}, untracked
        subprocess.run(['git', '-C', str(work), 'add', 'models/prompt.py', 'docs/code_changes.md'], check=True)
        subprocess.run(['git', '-C', str(work), '-c', 'user.name=LGSA experiment automation',
                        '-c', 'user.email=lgsa-experiment@localhost', 'commit',
                        '-m', 'Fix parent-child unpacking in HGS initialization'], check=True)
    verify_working(root)
    assert core.source_state(frozen) == original
    patch = git(work, 'diff', core.COMMIT, 'HEAD', '--', 'models/prompt.py') + '\n'
    print(patch, flush=True)
    print('PARENTMAP_PATCH_COMMITTED', git(work, 'rev-parse', 'HEAD'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=core.ROOT)
    args = parser.parse_args()
    baseline = json.loads((args.root / 'outputs/lgsa_official_smoke/frozen/2984968/SUCCESS.json').read_text())
    assert baseline['status'] == 'FROZEN_WOS_SMOKE_PASS'
    assert baseline['checkpoint_reload_logits_match'] and baseline['optimizer_steps'] == 64
    assert baseline['harness_sha256'] == core.sha(args.root / 'scripts/run_frozen_smoke.py')
    prepare(args.root)
