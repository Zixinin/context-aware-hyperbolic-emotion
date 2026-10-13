"""Verify original completed seed42 and all three matched Exp2 baselines."""
import argparse
import json
import sys
from m4lib import *
from seed_extension import seed_audit, derive_main, seed_environment

def check(work, out=None):
    validate_sources()
    environment = seed_environment(42)(False)
    success, pilot_meta = verify_candidate(PILOT_RUN, 42)
    root = REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions'
    tree = REPO / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json'
    audits = {seed: seed_audit(seed, REPO, root, tree, BASELINES / f'seed{seed}') for seed in SEEDS}
    result = {'status': 'EXP4B_MULTI_SEED_PREFLIGHT_PASS', 'seeds': list(SEEDS),
        'seed42_reused_success_sha256': sha(PILOT_RUN / 'SUCCESS.json'),
        'pilot_code_sha256': PINNED_PILOT_PACKAGE,
        'extension_hashes': extension_hashes(),
        'environment': environment,
        'pilot_config': pilot_meta['config'],
        'paired_baselines': {str(seed): audits[seed][4] for seed in SEEDS},
        'seed_derivations': {str(seed): derive_main(seed)[1] for seed in (123, 2026)},
        'only_seed_changes_vs_completed_exp4b42': True,
        'hierarchy_modified': False, 'training_performed': False, 'work': str(Path(work).resolve())}
    if out:
        require(not Path(out).exists(), 'Preflight output already exists')
        write(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    check(a.work, a.out)
