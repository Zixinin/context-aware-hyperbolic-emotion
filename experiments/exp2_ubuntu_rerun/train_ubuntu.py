"""Ubuntu orchestration only; execute the byte-pinned original Exp2 train AST."""
import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from r2lib import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    p.add_argument('--phase', choices=['audit', 'smoke', 'full'])
    p.add_argument('--out', type=Path)
    p.add_argument('--unit-proof', type=Path)
    p.add_argument('--audit-run', type=Path)
    p.add_argument('--smoke-run', type=Path)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--verify-run', type=Path)
    a = p.parse_args()
    if a.verify_run:
        record = verify_run(a.verify_run)
        status = record['status']
        require(status in {'EXP2_UBUNTU_AUDIT_PASS', 'EXP2_UBUNTU_SMOKE_PASS', FULL_STATUS, COMPLETE}, 'Unexpected completed stage')
        if status == FULL_STATUS:
            verify_full(a.verify_run, a.seed)
        elif status != COMPLETE:
            require(record['seed'] == a.seed and record['identity']['formal_config'] == formal_config(a.seed)
                and record['identity']['inputs']['rerun_code_sha256'] == code_hashes(), 'Stale/wrong completed gate')
        else:
            require(record['seeds'] == list(SEEDS) and record['all_seeds_verified'], 'Incomplete comparison')
        print('EXP2_UBUNTU_ARTIFACTS_VERIFIED', status, flush=True)
        return
    env = environment(a.seed, not a.preflight and a.phase != 'audit')
    if a.preflight:
        manifest, cfg, sources = audit(42, all_b4=True)
        result = {'status': 'EXP2_UBUNTU_PREFLIGHT_PASS', 'environment': env,
            'seeds': list(SEEDS), 'sources': sources, 'config_seed42': cfg,
            'new_gpu_training_performed': False, 'historical_results_modified': False}
        require(a.out is not None and not a.out.exists(), 'New preflight file required')
        write(a.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    require(a.phase and a.out and a.unit_proof, 'Need phase/out/unit-proof')
    unit = read(a.unit_proof)
    require(unit['status'] == 'EXP2_UBUNTU_UNIT_PASS' and unit['system'] == 'Linux'
        and unit['failures'] == unit['errors'] == 0 and unit['tests_run'] >= 17
        and unit['code_sha256'] == code_hashes(), 'Missing/stale actual Ubuntu unit proof')
    manifest, cfg, sources = audit(a.seed)
    identity = {'seed': a.seed, 'formal_config': cfg, 'inputs': sources,
        'environment': {k: v for k, v in env.items() if not k.startswith('gpu')},
        'unit_proof_sha256': sha(a.unit_proof)}
    if a.phase != 'audit':
        require(a.audit_run is not None, 'Audit gate missing')
        require(verify_run(a.audit_run, 'EXP2_UBUNTU_AUDIT_PASS')['identity'] == identity, 'Audit lineage changed')
    smoke = None
    if a.phase == 'full':
        check_committed()
        require(a.smoke_run is not None, 'Smoke gate missing')
        smoke = verify_run(a.smoke_run, 'EXP2_UBUNTU_SMOKE_PASS')
        require(smoke['identity'] == identity, 'Wrong/stale same-seed smoke')
    out = a.out.resolve()
    require(out.is_relative_to(REPO / 'results/exp2_ubuntu') and not out.exists(), 'New result directory only')
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out / 'config.json', cfg)
        write(out / 'environment.json', {**env, 'started_at': now(),
            'git_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()})
        with (out / 'requirements-installed.txt').open('x') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        source = (ROOT / 'goemotions_adaptation/run.py').read_text()
        _, ast_proof = compile_original(source, {})
        write(out / 'training_ast_identity.json', ast_proof)
        if a.phase == 'audit':
            write(out / 'SUCCESS.json', {'status': 'EXP2_UBUNTU_AUDIT_PASS', 'seed': a.seed,
                'identity': identity, 'training_steps': 0, 'artifacts_sha256': artifacts(out)})
            print('EXP2_UBUNTU_AUDIT_PASS', out, flush=True)
            return
        import torch
        from transformers import set_seed
        sys.path.insert(0, str(ROOT / 'goemotions_adaptation'))
        import run as legacy
        require(Path(legacy.__file__).resolve() == ROOT / 'goemotions_adaptation/run.py', 'Wrong legacy trainer')
        import comparison
        comparison.compare = lambda directory: None  # obsolete BERT42 report only
        run_cfg = dict(cfg)
        if a.phase == 'smoke':
            run_cfg.update(smoke_train=64, smoke_validation=32, smoke_epochs=1)
        legacy.config = lambda: run_cfg
        original_seed = legacy.seed
        def seed_all(value):
            original_seed(value); torch.cuda.manual_seed(value); set_seed(value)
        original_init = legacy.initialize_and_check
        calls = []
        def verify_and_initialize(model, tokenizer, hierarchy, source_path, actual_cfg, dest):
            proof = verify_raw(model, manifest)
            calls.append(proof)
            write(out / 'raw_initialization.json', proof)
            return original_init(model, tokenizer, hierarchy, source_path, actual_cfg, dest)
        namespace = dict(legacy.__dict__, config=lambda: run_cfg, seed=seed_all,
            initialize_and_check=verify_and_initialize)
        train, checked = compile_original(source, namespace)
        require(checked == ast_proof, 'Original training function drifted')
        meta = {'status': 'RUNNING', 'phase': a.phase, 'seed': a.seed,
            'experiment_name': 'Original Exp2 sentiment hierarchy Ubuntu rerun', 'config': cfg,
            'config_sha256': objsha(cfg), 'adapter_source': {'files_sha256': sources['source_records']},
            'dataset_manifest_sha256': sources['dataset_manifest_sha256'],
            'hierarchy_sha256': manifest['hierarchy_sha256'], 'hierarchy_depth': 2,
            'internal_supervision_nodes': 32, 'final_prediction_labels': NAMES,
            'identity': identity, 'test_used': False, 'start_time': now(),
            'runtime_smoke_overrides': {} if a.phase == 'full' else {k: run_cfg[k] for k in ('smoke_train', 'smoke_validation', 'smoke_epochs')},
            'trained_checkpoint_inherited': False, 'smoke_weights_inherited': False,
            'historical_hpc_results_modified': False}
        write(out / 'experiment_metadata.json', meta)
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        train(a.phase, out, manifest, run_cfg, meta, smoke, resume=False)
        meta.update(training_elapsed_seconds=time.monotonic()-began, end_time=now(),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), status='TRAINING_AND_EVALUATION_PASS')
        require(len(calls) == 1 and meta['optimizer_steps'] == (10856 if a.phase == 'full' else 4), 'Training update count differs')
        require(all(meta[k] for k in ('backbone_gradient_passed', 'gat_gradient_passed',
            'finite_training_passed', 'checkpoint_reload_logits_match')), 'Gradient/finite/reload evidence missing')
        b4_meta = read(B4_RUNS[a.seed] / 'experiment_metadata.json')
        require(meta['initial_backbone_sha256'] == b4_meta['initial_backbone_sha256'], 'Raw BERT fingerprint differs from paired Exp4B')
        if a.phase == 'full':
            require(meta['test_used'] and meta['test_examples'] == 5427, 'Full test evaluation incomplete')
            import numpy as np
            with np.load(out / 'test_predictions.npz', allow_pickle=False) as saved:
                scores, labels = metrics(saved['gold'], probabilities(saved['logits']), .30)
            write(out / 'test_fixed_0.30_complete_metrics.json', scores)
            csvwrite(out / 'test_fixed_0.30_complete_per_label.csv', labels)
        else:
            require(not meta['test_used'] and not (out / 'test_predictions.npz').exists(), 'Smoke used TEST')
        write(out / 'experiment_metadata.json', meta)
        require(audit(a.seed)[-1] == sources, 'Frozen inputs/source changed during execution')
        status = FULL_STATUS if a.phase == 'full' else 'EXP2_UBUNTU_SMOKE_PASS'
        write(out / 'SUCCESS.json', {'status': status, 'seed': a.seed, 'phase': a.phase,
            'identity': identity, 'initial_backbone_sha256': meta['initial_backbone_sha256'],
            'initial_model_sha256': meta['initial_model_sha256'], 'artifacts_sha256': artifacts(out),
            'completed_at': now()})
        if a.phase == 'full':
            verify_full(out, a.seed, manifest)
        else:
            verify_run(out, status)
        print(status, out, flush=True)
    except BaseException as error:
        write(out / 'FAILURE.json', {'status': 'EXP2_UBUNTU_STAGE_FAILED', 'error': repr(error), 'failed_at': now()})
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
