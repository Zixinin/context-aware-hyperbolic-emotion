"""Mechanical orchestration-only seed derivation; frozen training AST unchanged."""
import ast
import inspect
import subprocess
import sys
from pathlib import Path
from m4lib import *
import train as pilot_driver
import preflight as pilot_preflight

def function_tree(fn):
    return ast.parse(inspect.getsource(fn))

def derive_main(seed):
    require(seed in SEEDS, 'Unsupported seed')
    tree = function_tree(pilot_driver.main)
    changes = {'metadata_seed': 0, 'baseline_path': 0, 'experiment_title': 0, 'status': 0}
    class Replace(ast.NodeTransformer):
        def visit_Dict(self, node):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'seed' and isinstance(value, ast.Constant) and value.value == 42:
                    value.value = seed
                    changes['metadata_seed'] += 1
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if node.value == 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42':
                node.value = node.value[:-2] + str(seed)
                changes['baseline_path'] += 1
            elif node.value == 'Exp4B full data-driven hierarchy seed42':
                node.value = f'Exp4B full data-driven hierarchy seed{seed}'
                changes['experiment_title'] += 1
            elif node.value == 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED':
                if seed != 42:
                    node.value = 'EXP4B_SEED_FULL_COMPLETE_VERIFIED'
                changes['status'] += 1
            return node
    tree = ast.fix_missing_locations(Replace().visit(tree))
    require(changes == {'metadata_seed': 2, 'baseline_path': 1, 'experiment_title': 1, 'status': 1},
        'Frozen pilot orchestration drifted')
    return tree, changes

def compile_function(tree, namespace, name):
    namespace = dict(namespace)
    exec(compile(ast.fix_missing_locations(tree), '<Exp4B seed-only orchestration>', 'exec'), namespace)
    return namespace[name]

def seed_environment(seed):
    tree = function_tree(pilot_driver.environment)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == '42':
            node.value = str(seed)
            hits.append(node)
    require(len(hits) == 1, 'Frozen environment seed guard drifted')
    return compile_function(tree, pilot_driver.environment.__globals__, 'environment')

def paired_baseline(seed, baseline, root, manifest, expected):
    """Reuse every frozen baseline provenance/hash/array/checkpoint check."""
    require(seed in SEEDS and Path(baseline) == BASELINES / f'seed{seed}', 'Wrong matched baseline path')
    tree = function_tree(pilot_preflight.verify_baseline)
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 42:
            node.value = seed
            hits += 1
    require(hits == 2, 'Frozen baseline seed guard drifted')
    frozen_config = Path(root) / 'goemotions_adaptation/lgsa_goemotions_seed42.json'
    def seeded_read(path):
        value = read(path)
        return dict(value, seed=seed) if Path(path) == frozen_config else value
    ns = dict(pilot_preflight.verify_baseline.__globals__, read=seeded_read)
    proof = compile_function(tree, ns, 'verify_baseline')(baseline, root, manifest, expected)
    proof.update(status='EXP2_PAIRED_BASELINE_VERIFIED', paired_seed=seed)
    return proof

def seed_audit(seed, repo, root, tree_path, baseline):
    require(Path(repo) == REPO and seed in SEEDS, 'Wrong experiment context')
    validate_sources()
    h, manifest, cfg, comparison, _, sources = pilot_preflight.audit(
        repo, root, tree_path, BASELINES / 'seed42')
    expected = read(REPO / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
    proof = paired_baseline(seed, baseline, root, manifest, expected)
    reference = read(PILOT_RUN / 'config.json')
    require(cfg == reference, 'Completed seed42 formal configuration differs')
    seeded = dict(cfg, seed=seed)
    require([k for k in cfg if cfg[k] != seeded[k]] == ([] if seed == 42 else ['seed']), 'Non-seed hyperparameter changed')
    comparison.update(new_config=seeded, seed_change=seed, non_seed_changes_vs_completed_exp4b_seed42=[])
    sources = dict(sources, baseline_success_sha256=sha(Path(baseline) / 'SUCCESS.json'),
        pilot_seed42_success_sha256=sha(PILOT_RUN / 'SUCCESS.json'),
        extension_code_sha256=extension_hashes(), extension_protocol_sha256=sha(HERE / 'protocol.json'),
        matched_seed=seed, orchestration_derivation=derive_main(seed)[1])
    return h, manifest, seeded, comparison, proof, sources

def check_committed():
    pilot_driver.check_code_committed()
    relative = str(HERE.relative_to(REPO))
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
        '--untracked-files=all', '--', relative]).strip(), 'Freeze extension code before full training')
    tracked = set(subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '--', relative], text=True).splitlines())
    require({relative + '/' + n for n in extension_hashes()} <= tracked, 'Extension sources not tracked')

def paired_compare(seed):
    import compare_exp2_exp4b as original
    tree = function_tree(original.compare)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('seed42', f'seed{seed}').replace('SEED42', f'SEED{seed}')
            node.value = node.value.replace(f'NOT_RUN: current authorized scope is seed{seed} pilot',
                'SEED_ONLY: aggregate completion recorded by separate three-seed diagnostics')
            node.value = node.value.replace('Multi-seed and all pruning/flat-tree/peer-loss/hyperbolic ablations remain NOT RUN.',
                'This file is one paired seed. Separate diagnostics reports three-seed completion; all ablations remain NOT RUN.')
    return compile_function(tree, original.compare.__globals__, 'compare')

def main():
    import argparse
    import platform
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[123, 2026], required=True)
    p.add_argument('--extension-unit-proof', type=Path, required=True)
    args, remaining = p.parse_known_args()
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3'
        and PILOT.resolve() == REPO / 'experiments/exp4b_data_driven_hierarchy', 'Ubuntu workstation only')
    validate_sources()
    unit = read(args.extension_unit_proof)
    require(unit['status'] == 'EXP4B_EXTENSION_UNIT_PASS' and unit['system'] == 'Linux'
        and unit['tests_run'] >= 24 and unit['failures'] == unit['errors'] == 0
        and unit['extension_hashes'] == extension_hashes(), 'Missing/stale extension unit proof')
    if '--verify-run' in remaining:
        saved = read(Path(remaining[remaining.index('--verify-run') + 1]) / 'SUCCESS.json')
        require(saved.get('seed', saved['identity']['formal_config']['seed']) == args.seed and saved['identity']['inputs']['matched_seed'] == args.seed
            and saved['identity']['inputs']['extension_code_sha256'] == extension_hashes(), 'Completed stage seed/code lineage mismatch')
    import compare_exp2_exp4b as comparison
    comparison.compare = paired_compare(args.seed)
    namespace = dict(pilot_driver.main.__globals__, environment=seed_environment(args.seed),
        audit=lambda *a: seed_audit(args.seed, *a), check_code_committed=check_committed)
    sys.argv = [str(PILOT / 'train.py'), *remaining]
    compile_function(derive_main(args.seed)[0], namespace, 'main')()

if __name__ == '__main__':
    main()
