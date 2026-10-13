"""Independent Ubuntu rerun helpers. No legacy source/artifact is modified."""
import ast
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MULTI = HERE.parent / 'exp4b_multiseed_diagnostics'
if platform.system() != 'Linux':
    MULTI = Path(os.environ.get('EXP2_MULTI_SOURCE_DIR', MULTI)).resolve()
    REPO = Path(os.environ.get('EXP2_RERUN_REPO', REPO)).resolve()
sys.path.insert(0, str(MULTI))
import m4lib as multi
from m4lib import (NAMES, RARE, SIZES, SEEDS, FIELDS, objsha, sha, read, write,
    require, now, metrics, probabilities, csvwrite)
from seed_extension import seed_audit, seed_environment
from preflight import verify_raw

MULTI_PIN = 'bee48e734635f2f1c303f278fc68cbc61137f811eb36b4078b3f88b85a751695'
ROOT = REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions'
FROZEN = REPO / 'experiments/exp2_1_multiseed/frozen'
B4_WORK = REPO / 'results/exp4b/multiseed_20261013T043010Z'
B4_RUNS = {42: multi.PILOT_RUN, 123: B4_WORK / 'seed123/full', 2026: B4_WORK / 'seed2026/full'}
FULL_STATUS = 'EXP2_UBUNTU_SEED_FULL_COMPLETE_VERIFIED'
COMPLETE = 'EXP2_UBUNTU_RERUN_AND_EXP4B_COMPARISON_COMPLETE'

def code_hashes():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh'} and '__pycache__' not in p.parts}

def validate_sources():
    multi.validate_sources()
    require(objsha(multi.extension_hashes()) == MULTI_PIN, 'Frozen Exp4B multi-seed source differs')
    p = read(HERE / 'protocol.json')
    require(p['seeds'] == list(SEEDS) and p['text_window'] == 50
        and p['total_input_length'] == 54 and p['auxiliary_nodes'] == 32
        and p['depths'] == 2 and p['threshold'] == .30
        and not p['rerun_bert'] and not p['rerun_exp4b'] and not p['hyperparameter_tuning'],
        'Ubuntu rerun authorization differs')
    return p

def formal_config(seed):
    require(seed in SEEDS, 'Unsupported seed')
    cfg = read(FROZEN / 'lgsa.json')
    require(cfg['text_max_length'] == 50 and cfg['total_max_length'] == 54
        and cfg['fixed_threshold'] == .30 and cfg['epochs'] == 4, 'Frozen Exp2 config differs')
    return dict(cfg, seed=seed)

def compile_original(source, namespace):
    """Extract the original train function without even dimension replacements."""
    parsed = ast.parse(source)
    original = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    tree = ast.Module(body=[original], type_ignores=[])
    namespace = dict(namespace)
    exec(compile(ast.fix_missing_locations(tree), '<unchanged frozen Exp2 train>', 'exec'), namespace)
    proof = {'status': 'EXP2_ORIGINAL_TRAIN_AST_IDENTITY', 'training_ast_changes': 0,
        'original_train_ast_sha256': hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'only_runtime_hooks': ['requested seed', 'raw initialization verification', 'suppress obsolete BERT42 report']}
    return namespace['train'], proof

def verify_run(path, status=None):
    path = Path(path)
    record = read(path / 'SUCCESS.json')
    if status is not None:
        require(record['status'] == status, 'Wrong completed status')
    require(record['artifacts_sha256'], 'Empty artifact manifest')
    for name, wanted in record['artifacts_sha256'].items():
        target = path / name
        require(target.resolve().is_relative_to(path.resolve()) and sha(target) == wanted,
            'Artifact changed: ' + name)
    return record

def artifacts(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
        if p.is_file() and p.name not in {'SUCCESS.json', 'FAILURE.json'} and not p.name.endswith('.partial')}

def audit(seed, all_b4=False):
    validate_sources()
    old = seed_audit(seed, REPO, ROOT,
        REPO / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json',
        multi.BASELINES / f'seed{seed}')
    cfg = formal_config(seed)
    require(dict(cfg, total_max_length=68) == old[2], 'Non-hierarchy settings differ from completed Exp4B')
    b4_record = multi.verify_run(B4_WORK / 'diagnostics', 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE')
    require(b4_record['seeds'] == list(SEEDS) and b4_record['all_seeds_verified'], 'Exp4B three-seed completion missing')
    evidence = {}
    for s in SEEDS if all_b4 else (seed,):
        success, meta = multi.verify_candidate(B4_RUNS[s], s)
        require(meta['config'] == dict(formal_config(s), total_max_length=68), 'B4 non-hierarchy config differs')
        evidence[str(s)] = {'success_sha256': sha(B4_RUNS[s] / 'SUCCESS.json'),
            'initial_backbone_sha256': meta['initial_backbone_sha256']}
    inputs = dict(old[-1], rerun_code_sha256=code_hashes(),
        rerun_protocol_sha256=sha(HERE / 'protocol.json'),
        old_hierarchy_sha256=sha(ROOT / 'goemotions_adaptation/hierarchy.json'),
        completed_exp4b_diagnostics_sha256=sha(B4_WORK / 'diagnostics/SUCCESS.json'),
        paired_exp4b=evidence)
    return old[1], cfg, inputs

def environment(seed, need_gpu):
    env = seed_environment(seed)(need_gpu)
    previous = read(B4_RUNS[seed] / 'environment.json')
    require(all(env[k] == previous[k] for k in env if not k.startswith('gpu')),
        'Runtime defaults/software/host differ from completed Exp4B')
    if need_gpu:
        require(env['gpu'] == previous['gpu'], 'GPU model differs from completed Exp4B')
    return env

def check_committed():
    from train import check_code_committed as pilot_committed
    from seed_extension import check_committed as multi_committed
    pilot_committed(); multi_committed()
    relative = str(HERE.relative_to(REPO))
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
        '--untracked-files=all', '--', relative]).strip(), 'Freeze only new rerun sources first')
    tracked = set(subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '--', relative], text=True).splitlines())
    require({relative + '/' + n for n in code_hashes()} <= tracked, 'Rerun sources are not tracked')

def verify_full(path, seed, manifest=None):
    success = verify_run(path, FULL_STATUS)
    m = read(Path(path) / 'experiment_metadata.json')
    require(success['identity'] == m['identity'] and success['identity']['seed'] == seed
        and success['identity']['inputs']['rerun_code_sha256'] == code_hashes(), 'Completed rerun lineage/code differs')
    require(success['seed'] == m['seed'] == seed and m['phase'] == 'full'
        and m['config'] == formal_config(seed) and read(Path(path) / 'config.json') == m['config'], 'Rerun identity/config differs')
    require(m['optimizer_steps'] == 10856 and m['test_used'] is True and m['test_examples'] == 5427
        and m['train_examples'] == 43410 and m['validation_examples'] == 5426
        and m['checkpoint_reload_logits_match'] and m['finite_training_passed']
        and m['backbone_gradient_passed'] and m['gat_gradient_passed'], 'Full training evidence incomplete')
    history = [read(Path(path) / f'validation_epoch_{e}.json') for e in range(1, 5)]
    require(m['best_epoch'] == max(range(1, 5), key=lambda e: history[e-1]['macro_f1']), 'Wrong checkpoint selected')
    proof = read(Path(path) / 'raw_initialization.json')
    require(proof['tensors_compared'] == 197 and proof['pooler_present'] is False
        and len(set(proof['legacy_LayerNorm_aliases_verified'])) == 50
        and not proof['trained_checkpoint_loaded'] and not proof['smoke_weights_loaded'], 'LGSA raw backbone schema differs')
    if manifest is not None:
        for split in ('validation', 'test'):
            import numpy as np
            with np.load(Path(path) / (split + '_predictions.npz'), allow_pickle=False) as saved:
                y, z, ids = [saved[k].copy() for k in ('gold', 'logits', 'ids')]
            require(y.shape == z.shape == (SIZES[split], 28) and ids.shape == (SIZES[split],)
                and np.isfinite(z).all() and np.isin(y, [0, 1]).all(), 'Rerun prediction dimensions/values differ')
            sig = manifest['dataset_signatures'][split]
            require(objsha(ids.tolist()) == sig['ids_sha256']
                and hashlib.sha256(y.astype('uint8').tobytes()).hexdigest() == sig['gold_sha256'], 'Rerun IDs/gold/order differ')
    return success, m
