"""Separate extension helpers; distinct module name avoids legacy common collision."""
from pathlib import Path
import os
import platform
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PILOT = HERE.parent / 'exp4b_data_driven_hierarchy'
if platform.system() != 'Linux':
    PILOT = Path(os.environ.get('EXP4B_PILOT_SOURCE_DIR', PILOT)).resolve()
sys.path.insert(0, str(PILOT))
from b4lib import (NAMES, RARE, SIZES, TREE_SHA, REVISION, objsha, sha, read,
    write, require, now, package_hashes)
from train import verify_run
from evaluate import metrics, probabilities, csvwrite

PINNED_PILOT_PACKAGE = 'ac43c7531186856af2c7f5e409ac7735a68b9808d11d4b05187a3dc610beb1a5'
SEEDS = (42, 123, 2026)
FIELDS = ('micro_f1', 'macro_f1', 'weighted_f1', 'macro_precision',
    'macro_recall', 'rare_mean_f1', 'nonrare_mean_f1')
FOCUS = ('pride', 'grief', 'relief', 'nervousness', 'realization',
    'remorse', 'excitement', 'sadness')
PILOT_RUN = REPO / 'results/exp4b/run_20261012T130406Z/seed42'
BASELINES = REPO / 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions'

def extension_hashes():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh'}
        and '__pycache__' not in p.parts}

def validate_sources():
    require(objsha(package_hashes()) == PINNED_PILOT_PACKAGE, 'Frozen37-test pilot source differs; do not overwrite it')
    p = read(HERE / 'protocol.json')
    require(p['seeds'] == list(SEEDS) and p['new_training_seeds'] == [123, 2026]
        and p['threshold'] == .30 and p['text_window'] == 50
        and p['total_input_length'] == 68 and p['hierarchy_sha256'] == TREE_SHA
        and not any(p[k] for k in ('pruning', 'threshold_tuning', 'peer_loss', 'hyperbolic', 'other_ablations')),
        'Extension authorization changed')
    return p

def candidate_paths(work):
    return {42: PILOT_RUN, 123: Path(work) / 'seed123/full', 2026: Path(work) / 'seed2026/full'}

def verify_candidate(path, seed):
    status = 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED' if seed == 42 else 'EXP4B_SEED_FULL_COMPLETE_VERIFIED'
    success = verify_run(path, status)
    m = read(Path(path) / 'experiment_metadata.json')
    require(success['seed'] == m['seed'] == m['config']['seed'] == seed,
        'Candidate seed provenance mismatch')
    require(success['phase'] == m['phase'] == 'full' and m['optimizer_steps'] == 10856
        and m['test_used'] is True and m['test_examples'] == 5427, 'Candidate full run incomplete')
    return success, m

def artifact_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
        if p.is_file() and p.name not in {'SUCCESS.json', 'FAILURE.json'}
        and not p.name.endswith('.partial')}
