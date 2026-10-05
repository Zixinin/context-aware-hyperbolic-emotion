"""Independent LGSA GoEmotions smoke/full processes; full never inherits smoke weights."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import traceback

import numpy as np
import torch
from common import ROOT, PACKAGE, NAMES, UPSTREAM, PATCH_COMMIT, UPSTREAM_JOB, require, read, write, sha, now, config, source_record
from prepare import ASSETS, verify_assets
from hierarchy_goemotions import load_hierarchy
from data_goemotions import rows, sample, tensors, TensorRows
from evaluation import metrics, select_threshold, save_report, plots
from geometry import initialize_and_check, describe
from gates import upstream_gate, smoke_gate


def atomic_save(path, payload):
    temp = path.with_name(path.name + '.partial')
    torch.save(payload, temp)
    os.replace(temp, path)


def digest_state(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        t = tensor.detach().cpu().contiguous()
        h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def loader(data, batch, epoch=0, shuffle=False):
    order = list(range(len(data['input_ids'])))
    if shuffle:
        random.Random(config()['seed'] + epoch).shuffle(order)
    # DataLoader must not advance the training RNG when resuming within an epoch.
    return torch.utils.data.DataLoader(TensorRows(data, order), batch_size=batch, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(config()['seed'] + epoch))


@torch.no_grad()
def predict(model, data, hierarchy, cfg):
    model.eval()
    all_logits, all_gold = [], []
    for batch in loader(data, cfg['eval_batch_size']):
        _, all_depths = model.generate(batch['input_ids'].cuda(), depth2label=hierarchy['depths'])
        require(tuple(all_depths.shape[1:]) == (2, 32), 'Wrong LGSA output dimensions')
        logits = all_depths[:, 1, :28]
        require(bool(torch.isfinite(logits).all()), 'Nonfinite fine logits')
        all_logits.append(logits.cpu().numpy()); all_gold.append(batch['fine'].numpy())
    return np.concatenate(all_gold), np.concatenate(all_logits)


def train(phase, out, manifest, cfg, metadata, smoke, resume=False):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    import transformers
    import torch_geometric
    source = ROOT / 'working_lgsa'
    sys.path.insert(0, str(source))
    from models.prompt import Prompt
    from models.loss import HierarchicalGofLoss
    seed(cfg['seed'])
    torch.set_num_threads(4)
    hierarchy = load_hierarchy(PACKAGE / 'hierarchy.json')
    work = out / 'model_data'
    work.mkdir(exist_ok=True)
    torch.save(hierarchy['labels'], work / 'value_dict.pt')
    torch.save(hierarchy['slots'], work / 'slot.pt')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model']['path'], local_files_only=True,
                                             clean_up_tokenization_spaces=False)
    train_rows, val_rows = rows(ASSETS / 'train.jsonl'), rows(ASSETS / 'validation.jsonl')
    if phase == 'smoke':
        train_rows = sample(train_rows, cfg['smoke_train'], cfg['seed'])
        val_rows = sample(val_rows, cfg['smoke_validation'], cfg['seed'])
    train_data = tensors(train_rows, tokenizer, hierarchy, cfg['text_max_length'])
    val_data = tensors(val_rows, tokenizer, hierarchy, cfg['text_max_length'])
    write(out / 'sample_ids.json', {'train': [r['id'] for r in train_rows], 'validation': [r['id'] for r in val_rows]})
    model, loading = Prompt.from_pretrained(manifest['model']['path'], num_labels=32,
        path_list=hierarchy['edges'], layer=cfg['graph_layers'], graph_type=cfg['graph'],
        data_path=str(work), depth2label=hierarchy['depths'], local_files_only=True,
        output_loading_info=True, attn_implementation='eager')
    write(out / 'model_loading.json', loading)
    require(not any(k.startswith('bert.') for k in loading['missing_keys']), 'Missing raw BERT weights')
    require(not loading.get('mismatched_keys') and not loading.get('error_msgs'), 'Raw model compatibility failed')
    initial_backbone = digest_state(model.bert.state_dict())
    geometry = initialize_and_check(model, tokenizer, hierarchy, source, cfg, out)
    initial_model = digest_state(model.state_dict())
    if phase == 'full':
        require(initial_backbone == smoke['initial_backbone_sha256'], 'Full did not reset raw BERT')
        require(initial_model == smoke['initial_model_sha256'], 'Full initial LGSA state differs from fresh smoke initialization')
    metadata.update(initial_backbone_sha256=initial_backbone, initial_model_sha256=initial_model,
        initialization='raw_pretrained', initial_checkpoint=None, train_examples=len(train_rows),
        validation_examples=len(val_rows), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        transformers=transformers.__version__, torch_geometric=torch_geometric.__version__,
        gpu=torch.cuda.get_device_name(0), geometry=geometry, test_used=False,
        reproducibility='Seeded; CUDA scatter/GAT kernels may be nondeterministic; no bitwise reproducibility claim')
    write(out / 'experiment_metadata.json', metadata)
    print('RAW_CASED_INITIALIZATION_AND_HGS_PASS', phase, flush=True)
    model.cuda()
    require(model.get_input_embeddings().graph.path_list.max().item() == 33, 'Depth node indices incorrect')
    require(model.get_input_embeddings().size == tokenizer.vocab_size + 35, 'Graph/prompt vocabulary mismatch')
    gof = HierarchicalGofLoss(hierarchy['depths'], hierarchy['parents'], parent_child_angle_deg=cfg['gof_angle'],
                            length_decay_factor=cfg['gof_decay'], return_components=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                                 eps=cfg['epsilon'], weight_decay=cfg['weight_decay'])
    epochs = cfg['smoke_epochs'] if phase == 'smoke' else cfg['epochs']
    per_epoch = math.ceil(len(train_rows) / cfg['batch_size'])
    total = epochs * per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total*cfg['warmup_ratio'])), total)
    watched = model.bert.encoder.layer[0].attention.self.query.weight
    before = watched.detach().cpu().clone()
    step, first_epoch, first_batch, best, best_epoch = 0, 0, 0, -1.0, -1
    proof = {'backbone_gradient_passed': False, 'gat_gradient_passed': False, 'finite_training_passed': True}
    lineage = {'phase': phase, 'run_directory': str(out.resolve()), 'model': manifest['model'],
               'adapter_source': metadata['adapter_source'], 'config_sha256': metadata['config_sha256'],
               'dataset_manifest_sha256': metadata['dataset_manifest_sha256'], 'initial_model_sha256': initial_model}
    if resume:
        saved = torch.load(out / 'checkpoint_resume.pt', map_location='cpu', weights_only=False)
        require(saved['lineage'] == lineage and phase == 'full', 'Resume is restricted to this exact full run')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        step, first_epoch, first_batch = saved['step'], saved['epoch'], saved['next_batch']
        best, best_epoch, proof = saved['best'], saved['best_epoch'], saved['proof']
        restore_rng(saved['rng'])
        del saved
        # Remove log entries for updates not covered by the last durable checkpoint.
        records = [read_line for read_line in (out / 'steps.jsonl').read_text().splitlines()
                   if __import__('json').loads(read_line)['step'] <= step]
        (out / 'steps.jsonl').write_text('\n'.join(records) + ('\n' if records else ''))
        print('RESUME_SAME_FULL_RUN', step, first_epoch, first_batch, flush=True)

    def save_resume(epoch, next_batch):
        atomic_save(out / 'checkpoint_resume.pt', {'lineage': lineage, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step,
            'epoch': epoch, 'next_batch': next_batch, 'best': best, 'best_epoch': best_epoch,
            'proof': proof, 'rng': rng_state()})

    import json
    with (out / 'steps.jsonl').open('a' if resume else 'w') as log:
        for epoch in range(first_epoch, epochs):
            model.train()
            for batch_index, batch in enumerate(loader(train_data, cfg['batch_size'], epoch, shuffle=True)):
                if epoch == first_epoch and batch_index < first_batch:
                    continue
                batch.pop('fine')
                batch = {k: v.cuda() for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                output, mlm, classification = model(**batch)
                weights = model.get_input_embeddings().weight()[-35:-3]
                geo, components = gof(weights)
                alpha = min(1.0, step / max(1, int(total * cfg['hga_ramp_ratio'])))
                loss = output.loss + cfg['gof_loss_weight'] * alpha * geo
                require(bool(torch.isfinite(torch.stack([loss, mlm, classification, geo, *components.values()])).all()), 'Nonfinite training loss')
                loss.backward()
                if step == 1:
                    audit = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            require(bool(torch.isfinite(param.grad).all()), 'Nonfinite gradient: ' + name)
                            audit[name] = float(param.grad.norm())
                    proof['backbone_gradient_passed'] = watched.grad is not None and watched.grad.norm().item() > 0
                    proof['gat_gradient_passed'] = any('.graph.' in name and norm > 0 for name, norm in audit.items())
                    require(proof['backbone_gradient_passed'] and proof['gat_gradient_passed'], 'Missing backbone/GAT gradients')
                    write(out / 'gradient_audit.json', audit)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
                entry = {'step': step + 1, 'epoch': epoch + 1, 'batch': batch_index, 'lr': optimizer.param_groups[0]['lr'],
                         'alpha': alpha, 'loss': float(loss.detach()), 'mlm': float(mlm.detach()),
                         'classification': float(classification.detach()), 'hga': float(geo.detach()),
                         'grad_norm_before_clip': float(norm),
                         'hga_components_upstream': {k: float(v.detach()) for k, v in components.items()},
                         'sibling_family_mean': float((geo - components['global_orth'] - components['parent_child'] - components['len_decay']).detach())}
                optimizer.step(); scheduler.step(); step += 1
                log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
                if step % 20 == 0 or step == total:
                    print(f'{phase} epoch={epoch+1}/{epochs} step={step}/{total} loss={entry["loss"]:.5f}', flush=True)
                if phase == 'full' and (step % cfg['checkpoint_every_steps'] == 0 or batch_index + 1 == per_epoch):
                    save_resume(epoch, batch_index + 1)
            gold, logits = predict(model, val_data, hierarchy, cfg)
            score = metrics(gold, 1/(1+np.exp(-np.clip(logits,-80,80))), cfg['fixed_threshold'], manifest['rare_ids'])[0]
            write(out / f'validation_epoch_{epoch+1}.json', score)
            if score['macro_f1'] > best:
                best, best_epoch = score['macro_f1'], epoch + 1
                atomic_save(out / 'checkpoint_best.pt', {'phase': phase, 'lineage': lineage, 'model': model.state_dict(),
                                                       'epoch': best_epoch, 'score': score, 'step': step})
            print('VALIDATION', phase, epoch + 1, score, flush=True)
            if phase == 'full':
                save_resume(epoch + 1, 0)
    require(step == total and all(proof.values()), 'Training smoke/loop incomplete')
    delta = float((watched.detach().cpu() - before).abs().max())
    require(delta > 0 and math.isfinite(delta), 'Backbone never updated')
    # Load selected checkpoint, get reference logits, perturb, reload, and compare.
    chosen = torch.load(out / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    require(chosen['lineage'] == lineage, 'Best checkpoint lineage mismatch')
    model.load_state_dict(chosen['model'], strict=True)
    model.eval()
    probe_ids = val_data['input_ids'][:cfg['eval_batch_size']].cuda()
    with torch.no_grad():
        _, expected = model.generate(probe_ids, depth2label=hierarchy['depths'])
        model.multiclass_bias.add_(1)
        model.load_state_dict(chosen['model'], strict=True)
        _, actual = model.generate(probe_ids, depth2label=hierarchy['depths'])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    del chosen
    write(out / 'checkpoint_reload.json', {'status': 'CHECKPOINT_RELOAD_PASS', 'logits_match': True,
                                         'selected_epoch': best_epoch, 'selection': cfg['checkpoint_selection']})
    gold, logits = predict(model, val_data, hierarchy, cfg)
    np.savez_compressed(out / 'validation_predictions.npz', gold=gold, logits=logits, ids=np.array([r['id'] for r in val_rows]))
    save_report(out, 'validation_fixed_0.30', gold, logits, manifest['rare_ids'], cfg['fixed_threshold'])
    threshold = select_threshold(gold, 1/(1+np.exp(-np.clip(logits,-80,80))), cfg['threshold_candidates'], manifest['rare_ids'])
    write(out / 'threshold_selection.json', threshold)
    with torch.no_grad():
        learned = model.get_input_embeddings().weight()[-35:-3].cpu()
    torch.save({'learned_gat': learned}, out / 'learned_geometry.pt')
    write(out / 'learned_geometry.json', describe(learned, hierarchy))
    plots(out, hierarchy, learned)
    if phase == 'full':
        test_rows = rows(ASSETS / 'test.jsonl')
        test_data = tensors(test_rows, tokenizer, hierarchy, cfg['text_max_length'])
        y, z = predict(model, test_data, hierarchy, cfg)
        np.savez_compressed(out / 'test_predictions.npz', gold=y, logits=z, ids=np.array([r['id'] for r in test_rows]))
        fixed = save_report(out, 'test_fixed_0.30', y, z, manifest['rare_ids'], cfg['fixed_threshold'])
        selected = save_report(out, 'test_validation_selected', y, z, manifest['rare_ids'], threshold['selected_threshold'])
        native, _ = metrics(y, (z > 0).astype(float), 0.5, manifest['rare_ids'])
        write(out / 'test_native_logit_gt0.json', native)
        metadata.update(test_used=True, test_examples=len(test_rows), test_fixed=fixed, test_validation_selected=selected)
        from comparison import compare
        compare(out)
    metadata.update(optimizer_steps=step, best_epoch=best_epoch, backbone_parameter_max_change=delta,
                    checkpoint_reload_logits_match=True, **proof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'full'], required=True)
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--resume-run', type=Path)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    def interrupted(signum, frame):
        raise SystemExit('Interrupted; resume from the last durable full-run checkpoint')
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    require(job and job.isdecimal() and platform.system() == 'Linux' and torch.cuda.is_available(), 'Real training requires Slurm Linux GPU')
    require(torch.__version__ == '2.5.1+cu118' and torch.version.cuda == '11.8', 'Unexpected Torch/CUDA environment')
    upstream_gate()
    manifest, cfg, source = verify_assets(), config(), source_record()
    smoke = None
    if args.phase == 'full':
        require(args.smoke_run is not None, 'Full requires independently verified smoke')
        smoke = smoke_gate(args.smoke_run.resolve(), manifest)
    else:
        require(args.resume_run is None, 'Only full supports checkpoint resume')
    parent = ROOT / 'outputs/lgsa_goemotions' / ('smoke' if args.phase == 'smoke' else 'seed42')
    out = parent / job
    attempts = []
    if args.resume_run:
        out = args.resume_run.resolve()
        require(out.parent == parent.resolve() and not (out / 'SUCCESS.json').exists(), 'Resume must target an unfinished full run')
        previous = read(out / 'experiment_metadata.json')
        require(previous['phase'] == 'full', 'Never resume from a smoke checkpoint')
        attempts = previous.get('attempts', [])
    else:
        out.mkdir(parents=True, exist_ok=False)
    attempts.append({'job_id': job, 'start_time': now(), 'resume': bool(args.resume_run)})
    metadata = {'status': 'RUNNING', 'phase': args.phase, 'experiment_name': cfg['experiment_name'],
        'slurm_job_id': job, 'hostname': platform.node(), 'python': platform.python_version(), 'start_time': now(),
        'original_LGSA_commit': UPSTREAM, 'tested_parentmap_commit': PATCH_COMMIT, 'upstream_job_id': UPSTREAM_JOB,
        'adapter_source': source, 'config': cfg, 'config_sha256': sha(PACKAGE / 'lgsa_goemotions_seed42.json'),
        'dataset_manifest_sha256': sha(ASSETS / 'manifest.json'), 'model': manifest['model'],
        'hierarchy_sha256': manifest['hierarchy_sha256'], 'hierarchy_name': 'GoEmotions sentiment-derived hierarchy v0',
        'smoke_run': str(args.smoke_run.resolve()) if args.smoke_run else None, 'attempts': attempts,
        'smoke_success_sha256': sha(args.smoke_run / 'SUCCESS.json') if args.smoke_run else None}
    write(out / 'experiment_metadata.json', metadata)
    try:
        with (out / 'requirements-installed.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        train(args.phase, out, manifest, cfg, metadata, smoke, bool(args.resume_run))
        require(source_record() == source, 'Adapter source changed during training')
        require(verify_assets() == manifest, 'Assets changed during training')
        upstream_gate()
        metadata['artifacts_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()
            and p.name not in {'experiment_metadata.json', 'SUCCESS.json', 'checkpoint_resume.pt', 'error.txt'}
            and not p.name.endswith('.partial')}
        metadata['status'] = 'GOEMOTIONS_SMOKE_PASS' if args.phase == 'smoke' else 'GOEMOTIONS_FULL_PASS'
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error))
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        metadata['end_time'] = now()
        write(out / 'experiment_metadata.json', metadata)
    write(out / 'SUCCESS.json', metadata)
    print(metadata['status'], out, flush=True)


if __name__ == '__main__':
    main()
