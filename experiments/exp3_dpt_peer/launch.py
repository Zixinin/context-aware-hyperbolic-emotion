"""Ubuntu-only, gated reuse of the unchanged Experiment 2 training loop."""
import argparse
import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import urllib.request

from protocol import (HERE, REPO, DPT_COMMIT, assets, bind_legacy, git_info,
                      package_hashes, read, require, sha, source_audit, verify_raw, write)
from baseline import verify_baseline


def now():
    return datetime.now(timezone.utc).isoformat()


def notify(message):
    url = os.environ.get('FYP_NTFY_URL')
    if url:
        try:
            urllib.request.urlopen(urllib.request.Request(url, data=message.encode()), timeout=5).close()
        except Exception:
            print('Notification unavailable; experiment status is unaffected', flush=True)


def same_gate(path, status, identity):
    proof = read(path)
    require(proof['status'] == status and proof['identity'] == identity, 'Stale/incompatible gate: ' + str(path))
    return proof


def regression(model, hierarchy, cfg, legacy, gof):
    import numpy as np
    import torch
    from lgsa_peer import LGSAWithPeer
    # Train/validation only; no test file is read by this stage.
    tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(model.name_or_path,
        local_files_only=True, clean_up_tokenization_spaces=False)
    records = legacy.rows(legacy.ASSETS / 'train.jsonl')[:2]
    data = legacy.tensors(records, tokenizer, hierarchy, cfg['text_max_length'])
    batch = {k: v.cuda() for k, v in data.items() if k != 'fine'}
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    ids_before = [id(p) for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                                 eps=cfg['epsilon'], weight_decay=cfg['weight_decay'])
    groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
    state_before = legacy.digest_state(model.state_dict())
    rng = legacy.rng_state()
    full_steps = cfg['epochs'] * math.ceil(43410 / cfg['batch_size'])
    alpha = min(1.0, 42 / max(1, int(full_steps * cfg['hga_ramp_ratio'])))
    def forward():
        output, mlm, cls = model(**batch)
        geo, _ = gof(model.get_input_embeddings().weight()[-35:-3])
        # Include nonzero HGA weight to compare the complete original objective.
        total = output.loss + cfg['gof_loss_weight'] * alpha * geo
        gradient = torch.autograd.grad(total, parameters, allow_unused=True)
        mask = batch['input_ids'] == (model.get_input_embeddings().size - 1)
        logits = output.logits[mask].reshape(len(batch['input_ids']), 2, -1)[:, 1,
            model.vocab_size:model.vocab_size+28] + model.multiclass_bias[:28]
        logits = logits.detach().cpu().clone()
        values = torch.stack([output.loss, mlm, cls, geo, total]).detach().cpu()
        grad = [None if g is None else g.detach().cpu().clone() for g in gradient]
        return logits, values, grad, legacy.rng_state()
    old = forward()
    wrapped = LGSAWithPeer(model, hierarchy, {**cfg, 'peer_lambda': 0}).attach()
    legacy.restore_rng(rng)
    try:
        new = forward()
        torch.testing.assert_close(new[0], old[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new[1], old[1], rtol=1e-5, atol=1e-6)
        require(torch.equal(new[0].sigmoid() >= .30, old[0].sigmoid() >= .30), 'Predictions changed')
        for a, b in zip(old[2], new[2]):
            require((a is None) == (b is None), 'Gradient connectivity changed')
            if a is not None:
                torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
        require(old[3]['python'] == new[3]['python'], 'Python RNG changed')
        require(np.array_equal(old[3]['numpy'][1], new[3]['numpy'][1])
                and old[3]['numpy'][2:] == new[3]['numpy'][2:], 'NumPy RNG changed')
        require(torch.equal(old[3]['torch'], new[3]['torch']), 'Torch RNG changed')
        require(all(torch.equal(a, b) for a, b in zip(old[3]['cuda'], new[3]['cuda'])), 'CUDA RNG changed')
        require(ids_before == [id(p) for p in model.parameters()], 'Model parameters changed')
        require(groups == optimizer.state_dict()['param_groups'], 'Optimizer groups changed')
        require(state_before == legacy.digest_state(model.state_dict()), 'Regression mutated model state')
        return {'status': 'EXP2_LAMBDA_ZERO_REGRESSION_PASS', 'rtol': 1e-5, 'atol': 1e-6,
                'logits_match': True, 'original_losses_match': True, 'predictions_match': True,
                'gradients_match': True, 'rng_match': True, 'optimizer_groups_match': True,
                'hierarchy_and_tokenizer_unchanged': True, 'training_steps': 0}
    finally:
        wrapped.detach()
        legacy.restore_rng(rng)


def instrument_train(legacy, diagnostics):
    """Add ONLY scalar diagnostics after the original per-step entry assignment."""
    tree = ast.parse(inspect.getsource(legacy.train))
    class Insert(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'entry' for t in node.targets):
                self.count += 1
                return [node, ast.parse('entry.update(exp3_step_diagnostics(entry))').body[0]]
            return node
    transform = Insert()
    tree = transform.visit(tree)
    require(transform.count == 1, 'Original training loop logging structure changed')
    ast.fix_missing_locations(tree)
    namespace = dict(legacy.__dict__)
    namespace['exp3_step_diagnostics'] = diagnostics
    exec(compile(tree, str(HERE / 'derived_original_train.py'), 'exec'), namespace)
    return namespace['train']


class RegressionDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['audit', 'regression', 'smoke', 'full'], required=True)
    parser.add_argument('--lgsa-root', type=Path, required=True,
                        help='Existing Exp2 directory containing working_lgsa, goemotions_adaptation, assets')
    parser.add_argument('--model-path', type=Path, help='Relocated raw pinned snapshot; old manifests remain untouched')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unit-proof', type=Path)
    parser.add_argument('--upstream-proof', type=Path)
    parser.add_argument('--regression-run', type=Path)
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--baseline-run', type=Path, help='Validated Exp2 seed42 directory with predictions/metadata')
    args = parser.parse_args()
    import torch
    import transformers
    import numpy as np
    import torch_geometric
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3', 'Run on the specified Ubuntu workstation')
    require(sys.version_info[:2] == (3, 10), 'Use the existing validated Python3.10 environment')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible CUDA GPU required')
    require('4090' in torch.cuda.get_device_name(0), 'Expected RTX4090 GPU')
    free_memory, total_memory = torch.cuda.mem_get_info()
    require(free_memory >= 16 * 1024**3, 'Less than16GiB free GPU memory; inspect concurrent GPU work before proceeding')
    require(os.environ.get('PYTHONHASHSEED') == '42', 'Launch with PYTHONHASHSEED=42')
    require(torch.__version__ == '2.5.1+cu118' and torch.version.cuda == '11.8'
            and transformers.__version__ == '4.44.2' and np.__version__ == '1.26.4'
            and torch_geometric.__version__ == '2.6.1', 'Use the existing Exp2-compatible environment; do not upgrade')
    cfg = read(args.lgsa_root / 'goemotions_adaptation/lgsa_goemotions_seed42.json')
    cfg.update(read(HERE / 'config.json'))
    records = source_audit(args.lgsa_root)
    manifest = assets(args.lgsa_root, args.model_path)
    identity = {'package_sha256': package_hashes(), 'baseline_sources': records,
                'config': cfg, 'dataset_manifest_sha256': sha(args.lgsa_root / 'assets/goemotions-v0/manifest.json'),
                'raw_model_sha256': manifest['model']['sha256'], 'hostname': platform.node(),
                'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                    'transformers': transformers.__version__, 'numpy': np.__version__,
                    'torch_geometric': torch_geometric.__version__, 'cuda': torch.version.cuda}}
    out = args.out.resolve()
    require(out.is_relative_to(REPO / 'outputs/exp3_dpt_peer'), 'Output must be under new outputs/exp3_dpt_peer/')
    require(not out.exists(), 'Never overwrite an existing run')
    if args.phase != 'audit':
        require(args.unit_proof and args.upstream_proof, 'Unit and upstream proof files required')
        unit = read(args.unit_proof)
        require(unit['status'] == 'PEER_UNIT_TESTS_PASS' and unit['system'] == 'Linux'
                and unit['package_sha256'] == identity['package_sha256'], 'Stale/non-Ubuntu unit gate')
        upstream = read(args.upstream_proof)
        require(upstream['status'] == 'DPT_UPSTREAM_EXCERPT_PASS' and upstream['commit'] == DPT_COMMIT
                and upstream['excerpt_sha256'] == sha(HERE / 'upstream/dpt_sampling.py'), 'DPT provenance gate failed')
    if args.phase in ('smoke', 'full'):
        require(args.regression_run, 'Real lambda0 regression required')
        same_gate(args.regression_run / 'exp2_regression_check.json', 'EXP2_LAMBDA_ZERO_REGRESSION_PASS', identity)
    smoke = None
    if args.phase == 'full':
        package_status = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
            '--untracked-files=all', '--', 'experiments/exp3_dpt_peer'], text=True).strip()
        require(not package_status, 'Commit the validated Experiment3 code before full training')
        tracked = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '--',
            'experiments/exp3_dpt_peer'], text=True).splitlines()
        require(bool(tracked), 'Experiment3 package has not been committed')
        require(args.smoke_run and args.baseline_run, 'Smoke and existing Exp2 seed42 baseline required')
        smoke = same_gate(args.smoke_run / 'SUCCESS.json', 'EXP3_SMOKE_PASS', identity)
        for name, digest in smoke['artifacts_sha256'].items():
            require(sha(args.smoke_run / name) == digest, 'Smoke artifact changed: ' + name)
        baseline_proof = verify_baseline(args.baseline_run, args.lgsa_root, manifest)
        require(sha(args.lgsa_root / 'assets/goemotions-v0/test.jsonl') == manifest['files_sha256']['test.jsonl'],
                'Test data bytes changed')
    out.mkdir(parents=True, exist_ok=False)
    if args.phase == 'full':
        write(out / 'baseline_verification.json', baseline_proof)
    write(out / 'config.json', cfg)
    environment = {**identity['environment'], 'hostname': platform.node(), 'gpu': torch.cuda.get_device_name(0),
                   'gpu_free_memory_bytes_at_start': free_memory, 'gpu_total_memory_bytes': total_memory,
                   'python_executable': sys.executable, 'git': git_info(), 'started_at': now(),
                   'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    write(out / 'environment.json', environment)
    with (out / 'requirements-installed.txt').open('x') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    if args.phase == 'audit':
        write(out / 'AUDIT_PASS.json', {'status': 'EXP3_UBUNTU_AUDIT_PASS', 'identity': identity})
        print('EXP3_UBUNTU_AUDIT_PASS', out, flush=True)
        return
    legacy = bind_legacy(args.lgsa_root)
    legacy.config = lambda: cfg
    # Only suppress the obsolete BERT42 report; postprocessing compares LGSA42.
    import comparison
    comparison.compare = lambda directory: None
    original_init = legacy.initialize_and_check
    from lgsa_peer import LGSAWithPeer
    controller = {}
    metadata = {'status': 'RUNNING', 'phase': args.phase, 'experiment_name': cfg['experiment_name'],
                'start_time': now(), 'identity': identity, 'config': cfg,
                'config_sha256': sha(HERE / 'config.json'), 'adapter_source': {'files_sha256': records},
                'dataset_manifest_sha256': identity['dataset_manifest_sha256'], 'test_used': False}
    def initialize(model, tokenizer, hierarchy, source, original_cfg, destination):
        raw = verify_raw(model, manifest)
        write(out / 'raw_initialization.json', raw)
        geometry = original_init(model, tokenizer, hierarchy, source, original_cfg, destination)
        model.cuda()
        if args.phase == 'regression':
            from models.loss import HierarchicalGofLoss
            gof = HierarchicalGofLoss(hierarchy['depths'], hierarchy['parents'],
                parent_child_angle_deg=cfg['gof_angle'], length_decay_factor=cfg['gof_decay'], return_components=True).cuda()
            result = regression(model, hierarchy, cfg, legacy, gof)
            write(out / 'exp2_regression_check.json', {**result, 'identity': identity})
            raise RegressionDone()
        peer = LGSAWithPeer(model, hierarchy, cfg).attach()
        controller['peer'] = peer
        if args.phase == 'smoke':
            # Directly prove the NEW term has a nonzero finite encoder gradient.
            before_rng = legacy.rng_state()
            probe = legacy.tensors(legacy.rows(legacy.ASSETS / 'train.jsonl')[:16], tokenizer,
                                  hierarchy, cfg['text_max_length'])
            model.train()
            output, mlm, classification = model(**{k: v.cuda() for k, v in probe.items() if k != 'fine'})
            require(peer.last and peer.last['number_of_peer_terms'] > 0 and peer.last['peer_loss'] > 0,
                    'No eligible/nonzero peer probe')
            term = (output.loss - mlm - classification) / cfg['peer_lambda']
            watched = model.bert.encoder.layer[0].attention.self.query.weight
            gradient = torch.autograd.grad(term, watched)[0]
            require(bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0, 'Peer gradient disconnected')
            write(out / 'peer_gradient_probe.json', {'status': 'PEER_GRADIENT_PASS',
                'encoder_gradient_norm': float(gradient.norm()), **peer.last})
            legacy.restore_rng(before_rng)
            peer.records.clear()
            peer.last = None
        return geometry
    legacy.initialize_and_check = initialize
    def diagnostics(entry):
        peer = controller['peer']
        require(peer.last is not None, 'Peer forward did not execute')
        d = peer.last
        return {'original_lgsa_loss': d['original_forward_loss'] + cfg['gof_loss_weight'] * entry['alpha'] * entry['hga'],
                'peer_loss': d['peer_loss'], **{k: v for k, v in d.items() if k != 'original_forward_loss'}}
    train = instrument_train(legacy, diagnostics)
    notify('Experiment 3 ' + args.phase + ' started')
    try:
        try:
            train('full' if args.phase == 'full' else 'smoke', out, manifest, cfg, metadata, smoke, False)
        except RegressionDone:
            require(args.phase == 'regression', 'Unexpected regression intercept')
            print('EXP2_LAMBDA_ZERO_REGRESSION_PASS', out, flush=True)
            notify('Experiment 3 regression succeeded')
            return
        require(source_audit(args.lgsa_root) == records and package_hashes() == identity['package_sha256'],
                'Source changed during execution')
        require(assets(args.lgsa_root, args.model_path) == manifest, 'Assets changed during execution')
        from reporting import training_tables, compare_runs
        training_tables(out)
        if args.phase == 'full':
            compare_runs(args.baseline_run, out, args.lgsa_root / 'goemotions_adaptation/hierarchy.json')
        else:
            require(not (out / 'test_predictions.npz').exists(), 'Smoke read test data')
            steps = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
            require(any(r['number_of_peer_terms'] > 0 and r['peer_loss'] > 0 for r in steps), 'No peer training terms')
        metadata.update(status='EXP3_FULL_PASS' if args.phase == 'full' else 'EXP3_SMOKE_PASS', end_time=now())
        metadata['artifacts_sha256'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in {'SUCCESS.json', 'experiment_metadata.json'} and not p.name.endswith('.partial')}
        write(out / 'SUCCESS.json', metadata)
        print(metadata['status'], out, flush=True)
        notify('Experiment 3 ' + args.phase + ' succeeded')
    except BaseException as error:
        write(out / 'FAILURE.json', {'status': 'FAILED', 'phase': args.phase, 'error': str(error),
                                    'traceback': traceback.format_exc(), 'end_time': now()})
        notify('Experiment 3 ' + args.phase + ' failed')
        raise


if __name__ == '__main__':
    main()
