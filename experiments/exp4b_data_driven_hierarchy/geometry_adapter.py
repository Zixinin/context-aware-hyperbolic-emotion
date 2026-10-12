"""Generalize original HGS acceptance to all 56 nodes; keep initializer unchanged."""
import ast
from b4lib import require, write

def describe(vectors, hierarchy):
    import torch
    x = vectors.detach().cpu().double()
    require(x.shape[0] == hierarchy['num_nodes'] and bool(torch.isfinite(x).all()), 'Invalid hierarchy vectors')
    require(bool((x.norm(dim=1) > 1e-8).all()), 'Zero node prototype')
    def cosine(v):
        norms = v.norm(dim=1)
        require(bool((norms > 1e-8).all()), 'Degenerate sibling residual')
        unit = v / norms[:, None]
        matrix = unit @ unit.T
        return float(matrix[~torch.eye(len(v), dtype=torch.bool)].abs().max()) if len(v) > 1 else 0.0
    families = {}
    for parent, children in hierarchy['slots'].items():
        direction = x[parent] / x[parent].norm()
        residual = x[children] - (x[children] @ direction)[:, None] * direction
        families[str(parent)] = {'children': len(children), 'rank': int(torch.linalg.matrix_rank(residual)),
            'max_abs_cosine': cosine(residual), 'min_residual_norm': float(residual.norm(dim=1).min()),
            'sibling_constraint_applies': len(children) > 1}
    return {'top_max_abs_cosine': cosine(x[hierarchy['depths'][0]]), 'families': families,
        'min_label_norm': float(x.norm(dim=1).min()), 'max_label_norm': float(x.norm(dim=1).max())}

def initialize_and_check(model, tokenizer, hierarchy, source, cfg, out):
    import torch
    c = hierarchy['num_nodes']
    with torch.no_grad():
        initial = torch.stack([model.get_input_embeddings().weight[tokenizer.encode(hierarchy['labels'][i])].mean(0)
            for i in range(c)]).detach().cpu()
    model.init_embedding()
    structured = model.ideal_label_embeddings.detach().cpu().clone()
    tree = ast.parse((source / 'models/prompt.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'init_embedding')
    assignment = next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'value2slot' for t in n.targets))
    actual = eval(compile(ast.Expression(assignment.value), str(source / 'models/prompt.py'), 'eval'), {'self': model})
    actual.update({i: -1 for i in range(c) if i not in actual})
    require(actual == hierarchy['parents'], 'HGS parent mapping differs from complete tree')
    require(not torch.equal(initial[:28], structured[:28]), 'HGS did not affect fine prototypes')
    result = describe(structured, hierarchy)
    require(result['top_max_abs_cosine'] <= cfg['geometry_cosine_tolerance'], 'Top orthogonality failed')
    for family in result['families'].values():
        require(family['rank'] == family['children'], 'Rank-deficient sibling residuals')
        require(family['max_abs_cosine'] <= cfg['geometry_cosine_tolerance'], 'Sibling orthogonality failed')
    a, b = initial.double(), structured.double()
    tolerance = {'rtol': cfg['geometry_norm_rtol'], 'atol': cfg['geometry_norm_atol']}
    torch.testing.assert_close(a.norm(dim=1), b.norm(dim=1), **tolerance)
    for parent, children in hierarchy['slots'].items():
        direction = b[parent] / b[parent].norm()
        pa, pb = a[children] @ direction, b[children] @ direction
        torch.testing.assert_close(pa, pb, **tolerance)
        torch.testing.assert_close((a[children] - pa[:, None] * direction).norm(dim=1),
            (b[children] - pb[:, None] * direction).norm(dim=1), **tolerance)
    result.update(status='EXP4B_FULL_TREE_HGS_PASS', map_matches=True,
        all_supervision_nodes=c, fine_prediction_labels=28, norm_projection_preservation=True,
        initializer_source_unchanged=True, scope='Pre-GAT only; learned geometry is diagnostic')
    torch.save({'initial': initial, 'hgs': structured, 'parent_child_edges': hierarchy['edges']}, out / 'initial_geometry.pt')
    write(out / 'geometry_acceptance.json', result)
    return result

def plots(out, hierarchy, learned):
    # Original plotting helper assumes32 labels. Keep one numeric diagnostic instead
    # of replacing model logic or generating dozens of unnecessary figures.
    write(out / 'learned_geometry_full_tree.json', describe(learned, hierarchy))
