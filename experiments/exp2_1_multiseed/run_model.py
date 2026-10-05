"""Minimal runtime adapters around the unchanged, hash-verified original training loops.

Only seed, destination, independent raw-initialization evidence and reporting hooks
are supplied here. No optimizer, forward, backward, schedule or loss is replaced.
"""
import argparse
import gc
import hashlib
import os
import platform
import sys
import time
import traceback
from mslib import *


def digest(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_raw(state, path):
    import torch
    from safetensors import safe_open
    path = Path(path)
    require(path.name == REVISION, 'Initialization must use pinned RAW snapshot')
    manifest = read(PROJECTS['bert'] / 'assets/manifest.json')
    require(path.resolve() == Path(manifest['model_snapshot']).resolve(), 'Raw model path mismatch')
    require(sha(path / 'model.safetensors') == manifest['model_sha256']['model.safetensors'], 'Raw weight file hash mismatch')
    legacy_layernorm = []
    with safe_open(str(path / 'model.safetensors'), framework='pt', device='cpu') as raw:
        keys = set(raw.keys())
        for name, tensor in state.items():
            # The pinned cased checkpoint retains the older LayerNorm gamma/beta
            # names. Transformers renames these to weight/bias when loading.
            candidates = ['bert.' + name, name]
            raw_key = next((key for key in candidates if key in keys), None)
            if raw_key is None and name.endswith('.LayerNorm.weight'):
                old = name[:-len('weight')] + 'gamma'
                raw_key = next((key for key in ['bert.' + old, old] if key in keys), None)
            elif raw_key is None and name.endswith('.LayerNorm.bias'):
                old = name[:-len('bias')] + 'beta'
                raw_key = next((key for key in ['bert.' + old, old] if key in keys), None)
            require(raw_key is not None, 'Backbone tensor missing in RAW checkpoint: ' + name)
            reference = raw.get_tensor(raw_key)
            require(tensor.dtype == reference.dtype and torch.equal(tensor.detach().cpu(), reference), 'Backbone not raw: ' + name)
            if raw_key.endswith(('.LayerNorm.gamma', '.LayerNorm.beta')):
                legacy_layernorm.append(raw_key)
    return {'status': 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED', 'backbone_sha256': digest(state),
            'raw_model_revision': REVISION, 'raw_file_sha256': manifest['model_sha256']['model.safetensors'],
            'tensors_compared': len(state), 'legacy_layernorm_keys_verified': sorted(legacy_layernorm),
            'trained_checkpoint_loaded': False,
            'optimizer_state_loaded': False, 'smoke_weights_loaded': False}


class InitializationComplete(Exception):
    """Normal init-only exit, before optimizer construction or any forward pass."""


def run_bert(cfg, out):
    import torch
    sys.path.insert(0, str(PROJECTS['bert']))
    from src import train as legacy
    original_load = legacy.load_model
    proof = {}
    def load(path):
        tokenizer, model = original_load(path)
        if not proof:
            proof.update(verify_raw(model.bert.state_dict(), path))
            env = runtime_environment()
            require(env['deterministic_algorithms'] and env['cudnn_deterministic'] and not env['cudnn_benchmark'], 'BERT determinism drift')
            require(not env['matmul_tf32'] and not env['cudnn_tf32'], 'BERT TF32 drift')
            proof['runtime_environment'] = env
            write(out / 'raw_initialization.json', proof)
            print(proof['status'], flush=True)
        else:
            require(Path(path).resolve() == (out / 'checkpoints/best').resolve(), 'Unexpected post-training checkpoint')
        return tokenizer, model
    legacy.load_model = load
    # run() retains the exact original loop; the new shared smoke gate is external.
    legacy.run(cfg, out)
    require(bool(proof), 'BERT raw fingerprint hook did not run')
    return proof


def run_lgsa(cfg, phase, out, reference, identity):
    import torch
    sys.path.insert(0, str(PROJECTS['lgsa'] / 'goemotions_adaptation'))
    import run as legacy
    import comparison
    from transformers import set_seed
    from prepare import verify_assets
    from common import source_record
    original_seed = legacy.seed
    def seed_all(seed):
        original_seed(seed)
        torch.cuda.manual_seed(seed)
        set_seed(seed)  # repeats identical seeds, consumes no random samples; no new determinism flags
    legacy.seed = seed_all
    legacy.config = lambda: cfg  # loader() now receives the same requested seed as train().
    # Legacy compare() compares every output to BERT42; suppress only this obsolete
    # report. The final six-run aggregator performs correctly paired comparisons.
    comparison.compare = lambda directory: None
    manifest = verify_assets()
    source = source_record()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'RUNNING', 'phase': 'full' if phase == 'init' else phase,
        'experiment_name': 'Experiment 2.1 LGSA-GoEmotions', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'hostname': platform.node(), 'python': platform.python_version(), 'start_time': now(),
        'original_LGSA_commit': UPSTREAM, 'tested_parentmap_commit': PATCH, 'upstream_job_id': '2985243',
        'adapter_source': source, 'config': cfg, 'config_sha256': objsha(cfg),
        'dataset_manifest_sha256': sha(PROJECTS['lgsa'] / 'assets/goemotions-v0/manifest.json'),
        'model': manifest['model'], 'hierarchy_sha256': manifest['hierarchy_sha256'],
        'hierarchy_name': 'GoEmotions sentiment-derived hierarchy v0',
        'smoke_run': None, 'initialization_reference': str(reference) if reference else None,
        'orchestration_identity': identity}
    write(out / 'experiment_metadata.json', metadata)
    (out / 'requirements-installed.txt').write_text(capture([sys.executable, '-m', 'pip', 'freeze']) + '\n')
    proof, calls = {}, []
    original_digest = legacy.digest_state
    flags_before = runtime_environment()
    def check_digest(state):
        value = original_digest(state)
        calls.append(value)
        if len(calls) == 1:
            proof.update(verify_raw(state, manifest['model']['path']))
            env = runtime_environment()
            require(env['deterministic_algorithms'] == flags_before['deterministic_algorithms'] is False, 'LGSA deterministic algorithms changed')
            require(env['matmul_tf32'] == flags_before['matmul_tf32'] and env['cudnn_tf32'] == flags_before['cudnn_tf32'], 'LGSA TF32 defaults changed')
            require(env['cudnn_deterministic'] and not env['cudnn_benchmark'], 'LGSA cudnn setting drift')
            proof['runtime_environment'] = env
            write(out / 'raw_initialization.json', proof)
        elif len(calls) == 2:
            proof['initial_backbone_sha256'], proof['initial_model_sha256'] = calls
            proof.update(seed=cfg['seed'], config_sha256=objsha(cfg), reference_kind='raw_init_only_no_optimizer_no_training')
            write(out / 'raw_initialization.json', proof)
            if phase == 'init':
                write(out / 'INITIALIZATION_REFERENCE.json', {**proof, 'identity': identity,
                    'geometry_sha256': sha(out / 'geometry_acceptance.json')})
                raise InitializationComplete()
        return value
    legacy.digest_state = check_digest
    if phase == 'full':
        r = read(reference / 'INITIALIZATION_REFERENCE.json')
        require(r['seed'] == cfg['seed'] and r['config_sha256'] == objsha(cfg), 'Cross-seed initialization reference forbidden')
        require(r['identity'] == identity and r['reference_kind'] == 'raw_init_only_no_optimizer_no_training', 'Invalid reference lineage')
    else:
        r = None
    try:
        legacy.train('full' if phase == 'init' else phase, out, manifest, cfg, metadata, r, resume=False)
        require(phase != 'init', 'Initialization interception failed')
    except InitializationComplete:
        require(phase == 'init' and len(calls) == 2, 'Unexpected initialization exit')
        print('RAW_INITIALIZATION_REFERENCE_PASS (no training)', flush=True)
        return proof
    require(source_record() == source and verify_assets() == manifest, 'Original source/assets changed while training')
    metadata.update(status='GOEMOTIONS_FULL_PASS' if phase == 'full' else 'GOEMOTIONS_SMOKE_PASS', end_time=now())
    metadata['artifacts_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()
        and p.name not in {'experiment_metadata.json', 'SUCCESS.json', 'checkpoint_resume.pt'} and not p.name.endswith('.partial')}
    write(out / 'experiment_metadata.json', metadata)
    write(out / 'SUCCESS.json', metadata)
    return proof


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['bert', 'lgsa'], required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--phase', choices=['smoke', 'init', 'full'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--smoke-gate', type=Path)
    args = p.parse_args()
    import torch
    require(platform.system() == 'Linux' and os.environ.get('SLURM_JOB_ID', '').isdecimal() and torch.cuda.is_available(), 'Training requires Linux Slurm GPU')
    require(torch.cuda.device_count() == 1, 'Exactly one GPU required')
    require(os.environ.get('PYTHONHASHSEED') == str(args.seed), 'Set PYTHONHASHSEED before launching Python')
    audit = load_audit()
    current_sources(args.model)
    env = expected_environment(args.model)
    identity = {'release_sha256': release(), 'audit_sha256': sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'),
                'model': args.model, 'seed': args.seed, 'job_id': os.environ['SLURM_JOB_ID']}
    if args.phase == 'full':
        require(args.smoke_gate is not None, 'Full run requires the new infrastructure smoke gate')
        gate = read(args.smoke_gate)
        require(gate['status'] == 'INFRASTRUCTURE_SMOKE_PASS' and gate['release_sha256'] == identity['release_sha256']
                and gate['audit_sha256'] == identity['audit_sha256'] and gate['job_id'] == identity['job_id'], 'Invalid smoke gate')
        from evidence_ms import verify_legacy
        for model, r in gate['models'].items():
            require(sha(Path(r['run_directory']) / 'SUCCESS.json') == r['success_sha256'], 'Smoke success changed')
            verify_legacy(model, Path(r['run_directory']), seed=123, phase='smoke')
    cfg = config(args.model, args.seed, 'smoke' if args.phase == 'smoke' else 'full')
    require(not args.out.exists(), 'Never overwrite an existing run: ' + str(args.out))
    started, start_clock = now(), time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if args.model == 'bert':
        require(args.phase != 'init', 'BERT verifies raw tensors in its own load hook')
        proof = run_bert(cfg, args.out)
    else:
        proof = run_lgsa(cfg, args.phase, args.out, args.reference, identity)
    current_sources(args.model)
    require(identity['release_sha256'] == release(), 'Orchestration code changed during execution')
    if args.phase != 'init':
        from evidence_ms import verify_legacy
        checked = verify_legacy(args.model, args.out, args.seed, args.phase)
        write(args.out / 'RUN_PASS.json', {'status': 'RUN_PHASE_PASS', 'identity': identity, 'phase': args.phase,
            'started_at': started, 'ended_at': now(), 'elapsed_seconds': time.monotonic() - start_clock,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'gpu': torch.cuda.get_device_name(0),
            'number_of_gpus': 1, 'runtime_environment': proof['runtime_environment'], 'raw_initialization': proof,
            'legacy_evidence': checked, 'raw_initialization_sha256': sha(args.out / 'raw_initialization.json'),
            'slurm_completion': 'PENDING: finalizer must verify job and batch COMPLETED/0:0'})
        print('RUN_PHASE_PASS', args.model, args.seed, args.phase, args.out, flush=True)


if __name__ == '__main__':
    main()
