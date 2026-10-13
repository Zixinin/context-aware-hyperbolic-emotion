"""Fresh paired reports from saved Ubuntu predictions; no GPU inference."""
import argparse
import ast
import inspect
from r2lib import *
import diagnostics as frozen_diagnostics

REPORT_NAME = 'EXP2_UBUNTU_VS_EXP4B_SUMMARY.md'

def reporting_tree():
    tree = ast.parse(inspect.getsource(frozen_diagnostics.run))
    hits = {'filename': 0, 'status': 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY.md':
            node.value = REPORT_NAME; hits['filename'] += 1
        elif isinstance(node, ast.Constant) and node.value == 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE':
            node.value = COMPLETE; hits['status'] += 1
    require(hits == {'filename': 1, 'status': 1}, 'Frozen report aggregator drifted')
    return ast.fix_missing_locations(tree)

def render(*args):
    text = frozen_diagnostics.render_summary(*args)
    text = text.replace('# EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY', '# EXP2_UBUNTU_VS_EXP4B_SUMMARY')
    text = text.replace('完整 Exp4A hierarchy 保持不变。seed42 复用已完成结果，新增训练仅 seed123/2026。',
        'Exp2的42/123/2026均在Ubuntu重新训练；Exp4B三seed复用已完成结果，完整Exp4A hierarchy保持不变。历史HPC结果及报告未覆盖。')
    text = text.replace('读取原steps.jsonl，不改任何训练逻辑。', '本节读取现有Exp4B steps.jsonl，不改训练逻辑；新Exp2 losses另存loss_exp2_ubuntu_by_step.csv。')
    text = text.replace('Exp2来自历史HPC，Exp4B来自Ubuntu4090D。',
        '本报告Exp2和Exp4B均来自同一台D3020-A3 Ubuntu4090D；软件版本、runtime默认值和raw BERT指纹核对一致。CUDA/GAT可能非确定性，不能承诺逐位完全相同。')
    text = text.replace('三seed与只读诊断完成后停止，不自动启动其他实验。',
        '旧树Exp2为32监督节点/2深度/54tokens，新树Exp4B为56/9/68；text均50。总长度变化由hierarchy确定，非独立调参。三seed同机配对报告完成后停止，不运行新ablation。')
    return text

def run(work, out):
    baselines = work / 'baselines'
    def checked_audit(seed, repo, root, tree_path, baseline):
        require(Path(baseline).resolve() == (baselines / f'seed{seed}').resolve(), 'Wrong Ubuntu baseline')
        original = seed_audit(seed, REPO, ROOT, tree_path, multi.BASELINES / f'seed{seed}')
        _, m = verify_full(baseline, seed, original[1])
        import numpy as np
        with np.load(baseline / 'test_predictions.npz', allow_pickle=False) as saved:
            score = metrics(saved['gold'], probabilities(saved['logits']), .30)[0]
        recorded = read(baseline / 'test_fixed_0.30_complete_metrics.json')
        require(score == recorded, 'Saved Ubuntu metric cannot be reproduced')
        sources = dict(original[-1], ubuntu_exp2_success_sha256=sha(baseline / 'SUCCESS.json'),
            ubuntu_exp2_config= m['config'], comparison_source='Ubuntu rerun, not HPC',
            rerun_sources=code_hashes())
        return (*original[:4], {'status': 'EXP2_UBUNTU_PAIRED_BASELINE_VERIFIED',
            'recomputed_fixed_0_30_metrics': score, 'paired_seed': seed}, sources)
    def provenance_write(path, value):
        if Path(path).name == 'input_provenance.json':
            value = dict(value, seed42_retrained=False, exp4b_seeds_retrained=[],
                exp2_seeds_retrained=list(SEEDS), baseline_origin='Ubuntu',
                seed42_retrained_scope='Exp4B only; Exp2 seed42 WAS retrained')
        write(path, value)
    for seed in SEEDS:
        manifest = audit(seed)[0]
        verify_full(baselines / f'seed{seed}', seed, manifest)
        multi.verify_candidate(B4_RUNS[seed], seed)
        env = read(baselines / f'seed{seed}' / 'environment.json')
        b4env = read(B4_RUNS[seed] / 'environment.json')
        require(all(env[k] == b4env[k] for k in ('hostname', 'torch', 'cuda', 'transformers',
            'numpy', 'torch_geometric', 'gpu', 'deterministic_algorithms', 'matmul_tf32', 'cudnn_tf32')),
            'Completed run environments differ')
    namespace = dict(frozen_diagnostics.__dict__, BASELINES=baselines,
        candidate_paths=lambda unused: B4_RUNS, seed_audit=checked_audit,
        validate_sources=validate_sources, extension_hashes=code_hashes,
        render_summary=render, write=provenance_write, HERE=HERE, REPO=REPO)
    # Add original Exp2 losses before SUCCESS is created, without changing aggregation math.
    original_render = namespace['render_summary']
    def render_with_losses(*args):
        traces, receipts = [], []
        for seed in SEEDS:
            data, info = frozen_diagnostics.loss_trace(baselines / f'seed{seed}' / 'steps.jsonl', formal_config(seed))
            traces.extend(dict(seed=seed, **r) for r in data)
            receipts.append(dict(info, seed=seed, supervision_nodes=32,
                classification_scope='Original Exp2 classification loss over32 supervision nodes'))
        if traces:
            csvwrite(out / 'loss_exp2_ubuntu_by_step.csv', traces)
        write(out / 'loss_exp2_ubuntu_availability.json', receipts)
        hpc_rows = []
        import numpy as np
        for seed in SEEDS:
            pair = {}
            for origin, path in [('HPC', multi.BASELINES / f'seed{seed}'), ('Ubuntu', baselines / f'seed{seed}')]:
                with np.load(path / 'test_predictions.npz', allow_pickle=False) as saved:
                    pair[origin] = metrics(saved['gold'], probabilities(saved['logits']), .30)[0]
                hpc_rows.append(dict(seed=seed, origin=origin, **{k: pair[origin][k] for k in FIELDS}))
            hpc_rows.append(dict(seed=seed, origin='Ubuntu_minus_HPC',
                **{k: pair['Ubuntu'][k]-pair['HPC'][k] for k in FIELDS}))
        csvwrite(out / 'exp2_hpc_vs_ubuntu_metrics_by_seed.csv', hpc_rows)
        text = original_render(*args)
        text += '\n## 历史HPC对照保留\n\n'
        text += 'exp2_hpc_vs_ubuntu_metrics_by_seed.csv单独记录旧HPC与本次Ubuntu Exp2的差值。主表只用同机Ubuntu配对，不混合HPC结果。\n'
        return text
    namespace['render_summary'] = render_with_losses
    exec(compile(reporting_tree(), '<saved-prediction Ubuntu baseline report>', 'exec'), namespace)
    namespace['run'](work, out)
    verify_run(out, COMPLETE)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.work.resolve(), a.out.resolve())
