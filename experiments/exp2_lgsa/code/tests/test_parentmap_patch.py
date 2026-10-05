import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_parentmap_patch as preparation
import run_patched_smoke as runner


def actual_geometry(source):
    tree = ast.parse(source.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Prompt')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ['_gram_schmidt', 'semantically_anchored_gof_initialization']]
    cls = ast.ClassDef(name='Geometry', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), namespace)
    init = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'init_embedding')
    assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'value2slot' for t in n.targets))
    instance = namespace['Geometry']()
    instance.num_labels = 6
    instance.path_list = [(0, 2), (0, 3), (1, 4), (1, 5), (6, 0), (6, 1), (7, 2)]
    parents = eval(compile(ast.Expression(assignment.value), str(source), 'eval'), {'self': instance})
    parents.update({i: -1 for i in range(6) if i not in parents})
    return instance, parents


class PatchTests(unittest.TestCase):
    def test_only_model_direction_changes(self):
        hashes = preparation.verify_working(ROOT)
        self.assertIn('models/prompt.py', hashes)
        before = (ROOT / 'original_lgsa/models/prompt.py').read_bytes()
        after = (ROOT / 'working_lgsa/models/prompt.py').read_bytes()
        self.assertEqual(after, before.replace(preparation.OLD, preparation.NEW, 1))

    def test_actual_methods_and_acceptance_gate(self):
        geometry, parents = actual_geometry(ROOT / 'working_lgsa/models/prompt.py')
        raw_geometry, raw_parents = actual_geometry(ROOT / 'original_lgsa/models/prompt.py')
        self.assertEqual(parents, {0: -1, 1: -1, 2: 0, 3: 0, 4: 1, 5: 1})
        self.assertNotEqual(parents, raw_parents)
        torch.manual_seed(42)
        initial = torch.randn(6, 32)
        depths = {0: [0, 1], 1: [2, 3, 4, 5]}
        hgs = geometry.semantically_anchored_gof_initialization(initial, depths, parents)
        raw = raw_geometry.semantically_anchored_gof_initialization(initial, depths, raw_parents)
        self.assertTrue(torch.equal(initial[2:], raw[2:]))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            base, out = folder / 'frozen', folder / 'patched'
            base.mkdir()
            out.mkdir()
            edges = geometry.path_list
            torch.save({'initial': initial, 'hgs': raw, 'parent_child_edges': edges}, base / 'initial_geometry.pt')
            torch.save({'initial': initial, 'hgs': hgs, 'parent_child_edges': edges}, out / 'initial_geometry.pt')
            report = {'map_matches': True, 'fine_vectors_changed': True, 'expected_parent_map': parents,
                      'hgs': runner.core.geometry(hgs, depths[0], {0: [2, 3], 1: [4, 5]})}
            runner.core.write(out / 'hgs_diagnostics.json', report)
            with patch.object(runner, 'BASE', base):
                result = runner.geometry_acceptance(out)
                self.assertEqual(result['status'], 'PATCHED_HGS_GEOMETRY_PASS')
                report['hgs']['top_max_abs_cosine'] = 0.01
                runner.core.write(out / 'hgs_diagnostics.json', report)
                with self.assertRaises(AssertionError):
                    runner.geometry_acceptance(out)

    def test_integrity_override_does_not_recurse(self):
        original = runner.core.source_state
        frozen = ROOT / 'original_lgsa'
        def verify(path):
            if path.resolve() == frozen.resolve():
                return original(path)
            return preparation.verify_working(ROOT)
        with patch.object(runner.core, 'source_state', verify):
            self.assertIn('docs/code_changes.md', runner.core.source_state(ROOT / 'working_lgsa'))


if __name__ == '__main__':
    unittest.main()
