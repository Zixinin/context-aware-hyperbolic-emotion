#!/bin/bash
# Paste this whole block into the HPC terminal. No automatic upload.
(
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
mkdir -p scripts slurm logs
if [ -n "$(squeue -h -u "$USER" -n lgsa_wos_frozen)" ]; then
    echo "An lgsa_wos_frozen job is already queued/running; nothing submitted."
    squeue -u "$USER" -n lgsa_wos_frozen
    exit 1
fi
cat > scripts/run_frozen_smoke.py <<'LGSA_PY'
"""Bounded GPU engineering smoke using unmodified upstream modules and mapper."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions')
COMMIT = '4ea3dfb39dcee82654f443811471dd29be32e87e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_state(source):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    assert git('rev-parse', 'HEAD') == COMMIT
    assert not git('status', '--porcelain'), 'Frozen source is not clean'
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    return {name: sha(source / name) for name in names if name}


def preflight():
    source = ROOT / 'original_lgsa'
    hashes = source_state(source)
    data_root = ROOT / 'assets/wos-v2'
    data = json.loads((data_root / 'manifest.json').read_text())
    assert data['upstream_commit'] == COMMIT
    assert data['summary']['labels'] == 141
    for name, digest in data['files_sha256'].items():
        if name.startswith('smoke/'):
            assert sha(data_root / name) == digest, name
    model = json.loads((ROOT / 'assets/smoke_bert_uncased_manifest.json').read_text())
    assert model['repo_id'] == 'google-bert/bert-base-uncased'
    assert model['revision'] == '86b5e0934494bd15c9632b12f734a8a67f723594'
    for name, digest in model['model_sha256'].items():
        assert sha(Path(model['model_snapshot']) / name) == digest, name
    for filename in ['WebOfScience_train.json', 'WebOfScience_dev.json', 'slot.pt', 'value_dict.pt', 'wos.taxonomy', 'sample_ids.json']:
        assert (data_root / 'smoke/WebOfScience' / filename).is_file()
    assert json.loads((ROOT / 'environment/installation-cpu-ready.json').read_text())['status'] == 'LGSA_ENV_CPU_READY'
    return source, hashes, data_root, model


def original_mapper(source, tokenizer, count, depths):
    tree = ast.parse((source / 'train.py').read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'data_map_function']
    assert len(functions) == 1
    prefix = []
    for d in range(len(depths)):
        prefix.extend([tokenizer.vocab_size + count + d, tokenizer.vocab_size + count + len(depths)])
    prefix.append(tokenizer.sep_token_id)
    scope = dict(num_class=count, max_depth=len(depths), depth2label=depths, prefix=prefix)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source / 'train.py'), 'exec'), scope)
    return scope['data_map_function']


def geometry(tensor, tops, slots):
    import torch
    x = tensor.detach().cpu().double()
    assert torch.isfinite(x).all()
    assert (x.norm(dim=1) > 1e-8).all(), 'Degenerate label vector'
    def offdiag(v):
        norms = v.norm(dim=1)
        assert (norms > 1e-8).all(), 'Degenerate residual'
        unit = v / norms[:, None]
        cos = unit @ unit.T
        mask = ~torch.eye(len(v), dtype=torch.bool)
        return float(cos[mask].abs().max()) if len(v) > 1 else 0.0
    residuals = {}
    for p, children in slots.items():
        pvec = x[p]
        child = x[list(children)]
        residual = child - torch.outer(child @ pvec / (pvec @ pvec), pvec)
        residuals[str(p)] = {'max_abs_cosine': offdiag(residual),
                             'rank': int(torch.linalg.matrix_rank(residual)),
                             'children': len(children)}
    return {'top_max_abs_cosine': offdiag(x[tops]), 'sibling_residuals': residuals,
            'min_norm': float(x.norm(dim=1).min()), 'max_norm': float(x.norm(dim=1).max())}


def run(out, prepared):
    import torch
    import torch_geometric
    import transformers
    import datasets
    from torch_geometric.nn import GATConv
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    source, source_hashes, data_root, model_record = prepared
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert torch.__version__ == '2.5.1+cu118' and torch.version.cuda == '11.8'
    torch.set_num_threads(4)
    probe_x = torch.randn(5, 16, device='cuda', requires_grad=True)
    probe = GATConv(16, 16).cuda()
    probe_y = probe(probe_x, torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]], device='cuda'))
    probe_y.square().mean().backward()
    assert torch.isfinite(probe_y).all() and torch.isfinite(probe_x.grad).all()
    torch.cuda.synchronize()
    write(out / 'gpu_probe.json', {'status': 'GPU_PROBE_PASS', 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'pyg': torch_geometric.__version__,
        'transformers': transformers.__version__, 'datasets': datasets.__version__})
    del probe, probe_x, probe_y
    print('GPU_PROBE_PASS', flush=True)
    sys.path.insert(0, str(source))
    from models.prompt import Prompt
    from models.loss import HierarchicalGofLoss
    from eval import evaluate
    from train import get_exponential_with_warmup_scheduler
    from utils import seed_torch
    seed_torch(42)
    work = out / 'data/WebOfScience'
    shutil.copytree(data_root / 'smoke/WebOfScience', work)
    os.chdir(out)
    labels = torch.load(work / 'value_dict.pt', weights_only=False)
    slots = torch.load(work / 'slot.pt', weights_only=False)
    count = len(labels)
    parents = {c: p for p, children in slots.items() for c in children}
    edges = [(p, c) for c, p in parents.items()]
    parents.update({i: -1 for i in range(count) if i not in parents})
    depths = {0: [i for i in range(count) if parents[i] == -1],
              1: [i for i in range(count) if parents[i] != -1]}
    assert count == 141 and len(depths[0]) == 7
    for d, nodes in depths.items():
        edges.extend((count + d, i) for i in nodes)
    arch = model_record['model_snapshot']
    tokenizer = AutoTokenizer.from_pretrained(arch, local_files_only=True)
    mapper = original_mapper(source, tokenizer, count, depths)
    dataset = datasets.load_dataset('json', data_files={
        s: str(work / f'WebOfScience_{s}.json') for s in ['train', 'dev']},
        cache_dir=str(out / 'dataset_cache'))
    dataset = dataset.map(lambda batch: mapper(batch, tokenizer), batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    assert len(dataset['train']) == 256 and len(dataset['dev']) == 128
    batch_size = 4
    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=0)
    dev = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False, num_workers=0)
    model, loading = Prompt.from_pretrained(arch, num_labels=count, path_list=edges, layer=1,
        graph_type='GAT', data_path=str(work), depth2label=depths,
        local_files_only=True, output_loading_info=True)
    write(out / 'model_loading.json', loading)
    # Reject unexpected missing backbone weights; LGSA-specific tensors are new.
    assert not any(k.startswith('bert.') for k in loading['missing_keys']), loading
    assert not loading.get('mismatched_keys') and not loading.get('error_msgs'), loading
    with torch.no_grad():
        initial = torch.stack([model.get_input_embeddings().weight[tokenizer.encode(labels[i])].mean(0)
                               for i in range(count)]).detach().cpu()
    model.init_embedding()
    hgs = model.ideal_label_embeddings.detach().cpu().clone()
    prompt_tree = ast.parse((source / 'models/prompt.py').read_text())
    init = next(n for n in ast.walk(prompt_tree) if isinstance(n, ast.FunctionDef) and n.name == 'init_embedding')
    assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'value2slot' for t in n.targets))
    actual = eval(compile(ast.Expression(assignment.value), str(source / 'models/prompt.py'), 'eval'), {'self': model})
    actual.update({i: -1 for i in range(count) if i not in actual})
    write(out / 'hgs_diagnostics.json', {'expected_parent_map': parents, 'actual_parent_map': actual,
        'map_matches': actual == parents, 'fine_vectors_changed': not torch.equal(initial[depths[1]], hgs[depths[1]]),
        'initial': geometry(initial, depths[0], slots), 'hgs': geometry(hgs, depths[0], slots),
        'max_norm_change': float((hgs.norm(dim=1) - initial.norm(dim=1)).abs().max()),
        'note': 'Frozen geometry deviations are recorded, not silently fixed.'})
    torch.save({'initial': initial, 'hgs': hgs, 'parent_child_edges': edges}, out / 'initial_geometry.pt')
    model.cuda()
    embedding = model.get_input_embeddings()
    assert embedding.graph.path_list.min() >= 0 and embedding.graph.path_list.max() < count + len(depths)
    captures = []
    def capture(module, inputs, output):
        if len(captures) < 2:
            captures.append(output.detach().cpu())
    hook = embedding.graph.register_forward_hook(capture)
    gof = HierarchicalGofLoss(depths, parents, parent_child_angle_deg=10.0,
                              length_decay_factor=0.9, return_components=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    warmup_steps = 30070 // batch_size + 1
    scheduler = get_exponential_with_warmup_scheduler(optimizer, warmup_steps, warmup_steps * 50,
                                                       (1e-7 / 1e-4) ** (1 / 49))
    watched = model.bert.encoder.layer[0].attention.self.query.weight
    before = watched.detach().cpu().clone()
    grad_report = None
    model.train()
    with (out / 'steps.jsonl').open('w') as log:
        for step, batch in enumerate(train):
            batch = {k: v.cuda() for k, v in batch.items()}
            assert batch['input_ids'].shape[1] == 512 and batch['labels'].shape[1] == count * 2
            output, mlm, classification = model(**batch)
            label_weights = embedding.weight()[-count - len(depths) - 1:-len(depths) - 1]
            geo, components = gof(label_weights)
            alpha = min(1.0, step / max(1, int(0.1 * len(train))))
            loss = output.loss + alpha * geo
            assert torch.isfinite(torch.stack([loss, mlm, classification, geo, *components.values()])).all()
            loss.backward()
            if step == 1:
                grad_report = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                    grad_report[name] = {'trainable': parameter.requires_grad,
                        'grad_norm': None if parameter.grad is None else float(parameter.grad.norm())}
                assert watched.grad is not None and watched.grad.norm() > 0
                assert any('.graph.' in n and (r['grad_norm'] or 0) > 0 for n, r in grad_report.items())
                write(out / 'gradient_audit.json', grad_report)
            entry = {'step': step, 'lr_used': optimizer.param_groups[0]['lr'], 'alpha': alpha,
                     'total': float(loss.detach()), 'mlm': float(mlm.detach()),
                     'classification': float(classification.detach()), 'hga': float(geo.detach()),
                     'hga_components_upstream': {k: float(v.detach()) for k, v in components.items()},
                     'brother_res_family_mean': float((geo - components['global_orth'] - components['parent_child'] - components['len_decay']).detach())}
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            log.flush()
            if step == 0:
                hook.remove()
                assert len(captures) == 2
                torch.save({'forward_gat': captures[0], 'hga_gat': captures[1]}, out / 'first_step_gat.pt')
            print(f'STEP {step + 1}/{len(train)} loss={entry["total"]:.5f} alpha={alpha:.3f}', flush=True)
    assert grad_report is not None
    delta = float((watched.detach().cpu() - before).abs().max())
    assert delta > 0, 'No observed backbone parameter update'
    model.eval()
    predictions, gold = [], []
    first_input = None
    first_logits = None
    with torch.no_grad():
        for batch in dev:
            batch = {k: v.cuda() for k, v in batch.items()}
            pred, logits = model.generate(batch['input_ids'], depth2label=depths)
            assert torch.isfinite(logits).all()
            if first_input is None:
                first_input, first_logits = batch['input_ids'].clone(), logits.detach().cpu()
            predictions.extend([list(set(p)) for p in pred])
            gold.extend([torch.where(g.view(2, count).eq(1).any(dim=0))[0].cpu().tolist() for g in batch['labels']])
        final_gat = embedding.weight()[-count - len(depths) - 1:-len(depths) - 1].cpu()
    scores = evaluate(predictions, gold, labels)
    write(out / 'validation.json', scores)
    write(out / 'validation_predictions.json', {'predictions': predictions, 'gold': gold})
    torch.save({'trained_gat': final_gat}, out / 'trained_geometry.pt')
    write(out / 'trained_geometry.json', geometry(final_gat, depths[0], slots))
    checkpoint = out / 'checkpoint_last.pt'
    torch.save({'param': model.state_dict(), 'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'steps': len(train)}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with torch.no_grad():
        model.multiclass_bias.add_(1)
    model.load_state_dict(saved['param'], strict=True)
    with torch.no_grad():
        _, restored_logits = model.generate(first_input, depth2label=depths)
    torch.testing.assert_close(restored_logits.cpu(), first_logits, rtol=1e-5, atol=1e-6)
    assert source_state(source) == source_hashes
    return {'status': 'FROZEN_WOS_SMOKE_PASS', 'optimizer_steps': len(train),
            'backbone_parameter_max_change': delta, 'checkpoint_sha256': sha(checkpoint),
            'checkpoint_reload_logits_match': True, 'validation_micro_f1': scores['micro_f1'],
            'validation_macro_f1': scores['macro_f1'], 'geometry_correctness_claimed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    prepared = preflight()
    if args.preflight:
        print('FROZEN_SMOKE_PREFLIGHT_PASS')
        return
    job = os.environ.get('SLURM_JOB_ID')
    assert job and platform.system() == 'Linux', 'Submit through Slurm on HPC'
    out = ROOT / 'outputs/lgsa_official_smoke/frozen' / job
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'experiment_name': 'frozen_WoS_engineering_smoke', 'status': 'RUNNING',
        'original_LGSA_commit': COMMIT, 'adaptation_commit': None, 'slurm_job_id': job,
        'hostname': platform.node(), 'python': platform.python_version(),
        'start_time': datetime.now(timezone.utc).isoformat(), 'seed': 42, 'batch_size': 4,
        'epochs': 1, 'learning_rate': 1e-4, 'max_length': 512, 'precision': 'float32',
        'gof_loss_weight': 1.0, 'gof_angle': 10.0, 'gof_decay': 0.9,
        'threshold_protocol': 'upstream generate: logit > 0 per depth',
        'model': prepared[3], 'dataset_manifest_sha256': sha(prepared[2] / 'manifest.json'),
        'hierarchy_sha256': sha(prepared[2] / 'smoke/WebOfScience/slot.pt'),
        'harness_sha256': sha(__file__), 'source_sha256': prepared[1],
        'execution_notes': ['Unmodified upstream model/loss/evaluate and AST-extracted data_map_function.',
            'Independent bounded loop: one epoch, no test split, always save/reload last checkpoint.',
            'Alpha uses explicit batch index, not tqdm refresh-dependent p_bar.n.',
            'Original WoS LR warmup of 7518 steps retained; not a quality benchmark.',
            'Upstream brother_res component reports last family; family mean also recorded.']}
    write(out / 'experiment_metadata.json', metadata)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                       stdout=(out / 'requirements-installed.txt').open('w'))
        result = run(out, prepared)
        metadata.update(result)
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error))
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        metadata['end_time'] = datetime.now(timezone.utc).isoformat()
        write(out / 'experiment_metadata.json', metadata)
    write(out / 'SUCCESS.json', metadata)
    print('FROZEN_WOS_SMOKE_PASS', out, flush=True)


if __name__ == '__main__':
    main()
LGSA_PY
cat > slurm/lgsa_frozen_smoke.sbatch <<'LGSA_SLURM'
#!/bin/bash -l
#SBATCH --job-name=lgsa_wos_frozen
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
set -u
date --iso-8601=seconds
hostname
nvidia-smi
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/run_frozen_smoke.py
date --iso-8601=seconds
LGSA_SLURM
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
bash -n slurm/lgsa_frozen_smoke.sbatch
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/run_frozen_smoke.py --preflight
job_reply=$(sbatch --parsable slurm/lgsa_frozen_smoke.sbatch)
job_id=${job_reply%%;*}
printf '%s\n' "$job_id" > logs/frozen-smoke-last-jobid.txt
printf 'SUBMITTED_FROZEN_SMOKE_JOB=%s\n' "$job_id"
squeue -j "$job_id" -o "%.18i %.22j %.12T %.12M %R"
)
