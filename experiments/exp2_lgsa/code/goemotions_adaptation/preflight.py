"""CPU-only asset/source checks; a pending WoS job need not have finished yet."""
import os
import subprocess
import sys
from common import ROOT, PATCH_COMMIT, UPSTREAM, require, git, source_record, config, PACKAGE, read
from hierarchy_goemotions import load_hierarchy
from prepare import verify_assets


def main():
    require(sys.flags.optimize == 0, 'Unset PYTHONOPTIMIZE')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from prepare_parentmap_patch import verify_working
    verify_working(ROOT)
    require(git(ROOT / 'working_lgsa', 'rev-parse', 'HEAD') == PATCH_COMMIT, 'Expected tested HPC patch commit')
    require(git(ROOT / 'original_lgsa', 'rev-parse', 'HEAD') == UPSTREAM, 'Frozen source changed')
    require(source_record()['adaptation_commit'], 'Adapter commit required')
    submitted = ROOT / 'logs/goemotions-chain-submission.json'
    if submitted.exists():
        require(read(submitted)['adapter_source'] == source_record(), 'Adapter changed since submission')
    load_hierarchy(PACKAGE / 'hierarchy.json')
    verify_assets()
    cfg = config()
    require(cfg['total_max_length'] == cfg['text_max_length'] + 4, 'Prompt length mismatch')
    require(cfg['seed'] == 42 and cfg['epochs'] == 4 and cfg['batch_size'] == 16, 'Unexpected main protocol')
    require(cfg['graph'] == 'GAT' and cfg['precision'] == 'float32', 'Unexpected model protocol')
    actual = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    expected = (ROOT / 'outputs/lgsa_official_smoke/frozen/2984968/requirements-installed.txt').read_text()
    require(actual == expected, 'Python environment differs from tested WoS environment')
    print('GOEMOTIONS_CPU_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
