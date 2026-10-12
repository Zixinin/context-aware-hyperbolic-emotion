"""Ubuntu-only gated Exp4B launcher. Formal scope is seed42, not ablations."""
import argparse
import ast
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from b4lib import HERE, REPO, NAMES, now, objsha, package_hashes, read, require, sha, write
from preflight import audit, verify_raw
from loop_adapter import compile_train

def artifact_hashes(out):
    return {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in {'SUCCESS.json', 'FAILURE.json'}
        and not p.name.endswith('.partial')}

def verify_run(path, status):
    path = Path(path)
    record = read(path / 'SUCCESS.json')
    require(record['status'] == status, 'Wrong run status: ' + str(path))
    for name, wanted in record['artifacts_sha256'].items():
        p = path / name
        require(p.resolve().is_relative_to(path.resolve()) and sha(p) == wanted, 'Run artifact drift: ' + name)
    return record

def environment(need_gpu):
    import torch
    import transformers
    import torch_geometric
    import numpy as np
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3', 'Formal stages only run on D3020-A3 Ubuntu')
    require(sys.version_info[:2] == (3, 10), 'Use .venv-lgsa Python3.10, not base or another project env')
    require(Path(sys.executable).absolute() == REPO / '.venv-lgsa/bin/python', 'Use exact project .venv-lgsa/bin/python')
    require(os.environ.get('PYTHONHASHSEED') == '42', 'Set PYTHONHASHSEED=42 before Python starts')
    require(torch.__version__ == '2.5.1+cu118' and torch.version.cuda == '11.8'
        and transformers.__version__ == '4.44.2' and np.__version__ == '1.26.4'
        and torch_geometric.__version__ == '2.6.1', 'Frozen dependency versions changed; do not auto-upgrade')
    result = {'hostname': platform.node(), 'python': platform.python_version(), 'executable': sys.executable,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'numpy': np.__version__, 'torch_geometric': torch_geometric.__version__,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    if need_gpu:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need one visible CUDA GPU')
        require('4090' in torch.cuda.get_device_name(0), 'Expected RTX4090 GPU')
        free, total = torch.cuda.mem_get_info()
        require(free >= 16 * 1024**3, 'GPU busy: less than16GiB free; stop before model construction')
        result.update(gpu=torch.cuda.get_device_name(0), gpu_free_bytes=free, gpu_total_bytes=total)
    return result

def git_info():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--short', '--branch')}

def gate(path, status, identity):
    proof = verify_run(path, status)
    require(proof['identity'] == identity, 'Stale/incompatible stage gate')
    return proof

def check_code_committed():
    relative = 'experiments/exp4b_data_driven_hierarchy'
    result = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
        '--untracked-files=all', '--', relative], text=True).strip()
    require(not result, 'Commit validated Exp4B code before formal training; do not commit unrelated changes')
    tracked = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '--', relative], text=True).splitlines()
    require({relative + '/' + n for n in package_hashes()} <= set(tracked), 'Exp4B code is not fully tracked')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['audit', 'smoke', 'full'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--unit-proof', type=Path)
    parser.add_argument('--audit-run', type=Path)
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--verify-run', type=Path)
    args = parser.parse_args()
    if args.verify_run:
        result = read(args.verify_run / 'SUCCESS.json')
        verify_run(args.verify_run, result['status'])
        print('EXP4B_ARTIFACTS_VERIFIED', result['status'], flush=True)
        return
    require(args.phase and args.out and args.unit_proof, 'Need phase/out/unit-proof')
    env = environment(args.phase != 'audit')
    out = args.out.resolve()
    require(out.is_relative_to(REPO / 'results/exp4b') and out != REPO / 'results/exp4b', 'Outputs only under new results/exp4b/')
    require(not out.exists(), 'Never overwrite an existing stage directory')
    root = REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions'
    tree_path = REPO / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json'
    baseline = REPO / 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42'
    unit = read(args.unit_proof)
    require(unit['status'] == 'EXP4B_UNIT_TESTS_PASS' and unit['system'] == 'Linux'
        and unit['package_sha256'] == package_hashes(), 'Missing/stale Ubuntu unit proof')
    protocol = read(HERE / 'config/protocol.json')
    require(protocol['authorized_seed'] == 42 and protocol['text_window'] == 50
        and protocol['final_classifier_output_dimension'] == 28
        and protocol['new_total_input_length'] == 68
        and not any(protocol[k] for k in ('hierarchy_pruning', 'hierarchy_flattening',
            'manual_reconnection', 'peer_loss', 'other_ablations',
            'multi_seed_training_authorized_in_current_stage')), 'Authorized protocol changed')
    h, manifest, cfg, comparison, baseline_proof, sources = audit(REPO, root, tree_path, baseline)
    identity = {'package_sha256': package_hashes(), 'inputs': sources, 'formal_config': cfg,
        'environment': {k: v for k, v in env.items() if not k.startswith('gpu')},
        'unit_report_sha256': sha(args.unit_proof)}
    if args.phase != 'audit':
        require(args.audit_run, 'Audit gate required')
        gate(args.audit_run, 'EXP4B_UBUNTU_PREFLIGHT_PASS', identity)
    smoke = None
    if args.phase == 'full':
        check_code_committed()
        require(args.smoke_run, 'Independent smoke gate required')
        smoke = gate(args.smoke_run, 'EXP4B_SMOKE_TEST_PASS', identity)
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out / 'hierarchy_validation.json', {**h['summary'], 'hierarchy_sha256': sha(tree_path)})
        write(out / 'exp4b_config_comparison.json', comparison)
        write(out / 'baseline_verification.json', baseline_proof)
        write(out / 'config.json', cfg)
        write(out / 'environment.json', {**env, 'git': git_info(), 'started_at': now()})
        with (out / 'requirements-installed.txt').open('x') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        print(json.dumps(h['summary'], ensure_ascii=False, indent=2), flush=True)
        print('CONFIG_COMPARISON_PASS: text50, total54→68, final28, auxiliary32→56, depth2→9', flush=True)
        source = (root / 'goemotions_adaptation/run.py').read_text()
        from loop_adapter import derive
        tree, loop_proof, diff = derive(source, 56, 9)
        write(out / 'training_loop_derivation.json', loop_proof)
        with (out / 'derived_train.py').open('x') as stream:
            stream.write(ast.unparse(tree) + '\n')
        with (out / 'training_loop_dimension_only.diff').open('x') as stream:
            stream.write(diff)
        if args.phase == 'audit':
            write(out / 'SUCCESS.json', {'status': 'EXP4B_UBUNTU_PREFLIGHT_PASS', 'identity': identity,
                'artifacts_sha256': artifact_hashes(out), 'training_steps': 0, 'gpu_training': False})
            print('EXP4B_UBUNTU_PREFLIGHT_PASS', out, flush=True)
            return
        import torch
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(root / 'goemotions_adaptation'))
        import run as legacy
        require(Path(legacy.__file__).resolve() == root / 'goemotions_adaptation/run.py', 'Wrong trainer import')
        from transformers import set_seed
        from data_adapter import tensors
        from evaluate import predict, save_report
        from geometry_adapter import initialize_and_check, describe, plots
        import comparison as old_comparison
        old_comparison.compare = lambda directory: None  # obsolete report only; paired LGSA comparison below
        # The small smoke is diagnostic only. Formal config remains unchanged.
        run_cfg = dict(cfg)
        if args.phase == 'smoke':
            run_cfg.update(smoke_train=64, smoke_validation=32, smoke_epochs=1)
        legacy.config = lambda: run_cfg
        original_seed = legacy.seed
        def seed_all(value):
            original_seed(value)
            torch.cuda.manual_seed(value)
            set_seed(value)
        namespace = dict(legacy.__dict__)
        namespace.update(seed=seed_all, config=lambda: run_cfg,
            load_hierarchy=lambda ignored: h, tensors=tensors,
            predict=lambda model, data, hierarchy, cfg: predict(model, data, hierarchy, cfg, legacy.loader),
            describe=describe, plots=plots, save_report=save_report)
        raw_proofs = []
        def initialize(model, tokenizer, hierarchy, source, actual_cfg, destination):
            proof = verify_raw(model, manifest)  # CPU exact verification BEFORE HGS/CUDA/optimizer
            raw_proofs.append(proof)
            write(out / 'raw_initialization.json', proof)
            return initialize_and_check(model, tokenizer, hierarchy, source, actual_cfg, destination)
        namespace['initialize_and_check'] = initialize
        train, checked, _ = compile_train(source, namespace, h)
        require(checked == loop_proof, 'Derived training function changed')
        metadata = {'status': 'RUNNING', 'phase': args.phase, 'seed': 42,
            'experiment_name': 'Exp4B full data-driven hierarchy seed42', 'start_time': now(),
            'config': cfg, 'runtime_smoke_only_overrides': {} if args.phase == 'full' else
                {k: run_cfg[k] for k in ('smoke_train', 'smoke_validation', 'smoke_epochs')},
            'identity': identity, 'adapter_source': {'files_sha256': sources['source_records']},
            'config_sha256': objsha(cfg), 'dataset_manifest_sha256': sources['dataset_manifest_sha256'],
            'hierarchy_sha256': sources['hierarchy_sha256'], 'final_classifier_output_dimension': 28,
            'internal_hierarchy_supervision_dimension': 56, 'test_used': False}
        write(out / 'experiment_metadata.json', metadata)
        began = time.monotonic()
        train('full' if args.phase == 'full' else 'smoke', out, manifest, run_cfg, metadata, smoke, resume=False)
        metadata['training_elapsed_seconds'] = time.monotonic() - began
        expected_steps = 10856 if args.phase == 'full' else 4
        require(metadata['optimizer_steps'] == expected_steps and len(raw_proofs) == 1, 'Training loop incomplete')
        require(metadata['backbone_gradient_passed'] and metadata['gat_gradient_passed']
            and metadata['finite_training_passed'] and metadata['checkpoint_reload_logits_match'], 'Gradient/finite/reload checks failed')
        for name in ('validation_predictions.npz',) + (('test_predictions.npz',) if args.phase == 'full' else ()):
            import numpy as np
            with np.load(out / name, allow_pickle=False) as saved:
                require(saved['logits'].shape == saved['gold'].shape and saved['logits'].shape[1] == 28,
                    'Final classifier export is not N x 28')
        metadata.update(status='TRAINING_AND_EVALUATION_PASS', end_time=now(),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            task_output_labels=NAMES, trained_checkpoint_inherited=False,
            smoke_weights_inherited=False, experiment2_files_modified=False)
        if args.phase == 'smoke':
            require(metadata['test_used'] is False and not (out / 'test_predictions.npz').exists(), 'Smoke performed test inference')
            write(out / 'smoke_test_report.json', {'status': 'EXP4B_SMOKE_TEST_PASS',
                'initial_backbone_sha256': metadata['initial_backbone_sha256'],
                'initial_model_sha256': metadata['initial_model_sha256'],
                'final_classifier_output_dimension': 28, 'internal_supervision_nodes': 56,
                'num_depths': 9, 'text_window': 50, 'total_input_length': 68,
                'forward_backward_optimizer_steps': 4, 'test_inference_performed': False,
                'finite_losses_and_gradients': True, 'checkpoint_reload_verified': True,
                'elapsed_seconds': metadata['training_elapsed_seconds']})
            status = 'EXP4B_SMOKE_TEST_PASS'
        else:
            require(metadata['test_used'] is True and metadata['test_examples'] == 5427, 'Formal test evaluation missing')
            from compare_exp2_exp4b import compare
            compare(baseline, out, metadata)
            status = 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED'
        write(out / 'experiment_metadata.json', metadata)
        # Repeat read-only source/data/baseline audit after training, not just at launch.
        after = audit(REPO, root, tree_path, baseline)[-1]
        require(after == sources and package_hashes() == identity['package_sha256'], 'Old inputs or new code changed during execution')
        record = {'status': status, 'identity': identity,
            'initial_backbone_sha256': metadata['initial_backbone_sha256'],
            'initial_model_sha256': metadata['initial_model_sha256'],
            'baseline_success_sha256': sources['baseline_success_sha256'],
            'phase': args.phase, 'seed': 42, 'final_classifier_output_dimension': 28,
            'all_three_seeds_completed': False, 'ablations_run': False,
            'artifacts_sha256': artifact_hashes(out), 'completed_at': now()}
        write(out / 'SUCCESS.json', record)
        verify_run(out, status)
        print(status, out, flush=True)
    except BaseException as exc:
        write(out / 'FAILURE.json', {'status': 'EXP4B_STAGE_FAILED', 'phase': args.phase,
            'error_type': type(exc).__name__, 'error': str(exc), 'failed_at': now()})
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
