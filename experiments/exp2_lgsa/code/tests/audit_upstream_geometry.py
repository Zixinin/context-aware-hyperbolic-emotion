"""CPU-only probes of frozen upstream functions; NOT a model/GPU smoke test.

Extract the actual AST methods to avoid loading BERT, datasets, or PyG.
No model download, optimizer, training loop, or upstream file modification.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "original_lgsa/models/prompt.py"
tree = ast.parse(SOURCE.read_text())
prompt = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prompt")
methods = [n for n in prompt.body if isinstance(n, ast.FunctionDef) and n.name in {
    "_gram_schmidt", "semantically_anchored_gof_initialization"}]
cls = ast.ClassDef(name="Geometry", bases=[], keywords=[], body=methods, decorator_list=[])
graph_embedding = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraphEmbedding")
namespace = {"torch": torch, "nn": nn, "F": F}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls, graph_embedding], type_ignores=[])),
             str(SOURCE), "exec"), namespace)
geometry = namespace["Geometry"]()

# Extract the actual upstream map expression, not an independently written copy.
init = next(n for n in prompt.body if isinstance(n, ast.FunctionDef) and n.name == "init_embedding")
assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "value2slot" for t in n.targets))
edges = [(0, 2), (0, 3), (1, 4), (1, 5)]
self_stub = SimpleNamespace(path_list=edges, num_labels=6)
upstream_map = eval(compile(ast.Expression(assignment.value), str(SOURCE), "eval"), {"self": self_stub})
expected_map = {child: parent for parent, child in edges}
for i in range(6):
    upstream_map.setdefault(i, -1)
    expected_map.setdefault(i, -1)
assert upstream_map != expected_map
assert upstream_map[0] == 3 and expected_map[2] == 0

torch.manual_seed(42)
initial = torch.randn(6, 16, dtype=torch.float64)
depths = {0: [0, 1], 1: [2, 3, 4, 5]}
structured = geometry.semantically_anchored_gof_initialization(initial, depths, expected_map)
upstream_structured = geometry.semantically_anchored_gof_initialization(initial, depths, upstream_map)
assert abs(float(F.cosine_similarity(structured[0:1], structured[1:2]))) < 1e-12
assert torch.allclose(structured.norm(dim=1), initial.norm(dim=1), atol=1e-10)
residual_dots = []
for parent, children in [(0, [2, 3]), (1, [4, 5])]:
    p = structured[parent]
    kids = structured[children]
    residuals = kids - torch.outer(kids @ p / (p @ p), p)
    cosine = float(F.cosine_similarity(residuals[0:1], residuals[1:2]))
    assert abs(cosine) < 1e-12
    residual_dots.append(cosine)

# Rank deficiency can return fewer vectors than labels; zero set throws.
dependent = torch.tensor([[1., 0.], [2., 0.]])
rank = len(geometry._gram_schmidt(dependent))
assert rank == 1
try:
    geometry._gram_schmidt(torch.zeros(2, 3))
except RuntimeError:
    empty_basis_fails = True
else:
    empty_basis_fails = False
assert empty_basis_fails

# Actual GraphEmbedding constructor freezes the concatenated prototype rows.
ideal_parameter = nn.Parameter(torch.randn(6, 16))
label_prompt_mask = torch.cat([ideal_parameter, torch.randn(3, 16)], dim=0)
embedding = namespace["GraphEmbedding"](SimpleNamespace(pad_token_id=0, num_labels=6),
    nn.Embedding(20, 16), label_prompt_mask, graph_type="")
assert not embedding.new_embedding.weight.requires_grad
embedding.raw_weight().sum().backward()
assert ideal_parameter.grad is None

spec = importlib.util.spec_from_file_location("upstream_loss", ROOT / "original_lgsa/models/loss.py")
loss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_module)
current = torch.randn(6, 16, requires_grad=True)
loss, comps = loss_module.HierarchicalGofLoss(depths, expected_map, return_components=True)(current)
assert set(comps) == {"global_orth", "parent_child", "len_decay", "brother_res"}
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(current.grad).all()

report = {
    "scope": "CPU toy probes only; no BERT/PyG/CUDA/official-data execution",
    "torch_version": torch.__version__,
    "upstream_parent_map": upstream_map,
    "expected_parent_map": expected_map,
    "reversed_parent_map_confirmed": True,
    "correct_map_top_cosine": float(F.cosine_similarity(structured[0:1], structured[1:2])),
    "correct_map_sibling_residual_cosines": residual_dots,
    "upstream_map_changes_fine_vectors": not torch.equal(upstream_structured[2:], initial[2:]),
    "rank_deficient_input_rows": 2,
    "rank_deficient_output_rows": rank,
    "all_zero_basis_raises": empty_basis_fails,
    "prototype_rows_frozen": not embedding.new_embedding.weight.requires_grad,
    "ideal_parameter_receives_gradient_in_probe": ideal_parameter.grad is not None,
    "hga_four_components_and_backward_finite": True,
    "hga_components_sample": {k: float(v.detach()) for k, v in comps.items()},
    "hga_total_sample": float(loss.detach()),
    "warning": "brother_res reports the last family's loss, not the family mean used in total",
}
out = ROOT / "docs/upstream_geometry_audit.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
