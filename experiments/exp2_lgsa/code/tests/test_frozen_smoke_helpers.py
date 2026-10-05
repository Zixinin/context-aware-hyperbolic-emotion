import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke', ROOT / 'scripts/run_frozen_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class Tokenizer:
    vocab_size = 30522
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, truncation=True):
        ids = [101, 300, 400, 102] if text == 'short' else [101] + [300] * 600 + [102]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


class Helpers(unittest.TestCase):
    def test_exact_upstream_mapper_prompts_masks_and_truncation(self):
        tokenizer = Tokenizer()
        fn = smoke.original_mapper(ROOT / 'original_lgsa', tokenizer, 4, {0: [0, 1], 1: [2, 3]})
        mapped = fn({'token': ['short', 'long'], 'label': [[0, 2], [1, 3]]}, tokenizer)
        self.assertEqual(mapped['labels'], [[1, 0, -100, -100, -100, -100, 1, 0],
                                          [0, 1, -100, -100, -100, -100, 0, 1]])
        self.assertEqual([len(row) for row in mapped['input_ids']], [512, 512])
        self.assertEqual(mapped['input_ids'][0][3:8], [30526, 30528, 30527, 30528, 102])
        self.assertEqual(mapped['input_ids'][1][-5:], [30526, 30528, 30527, 30528, 102])
        self.assertEqual(sum(mapped['attention_mask'][0]), 8)
        self.assertEqual(sum(mapped['attention_mask'][1]), 512)

    def test_geometry_checks_residuals_not_whole_children(self):
        vectors = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [1, 0, 1., 0], [1, 0, 0, 1.]])
        report = smoke.geometry(vectors, [0, 1], {0: [2, 3]})
        self.assertEqual(report['top_max_abs_cosine'], 0)
        self.assertEqual(report['sibling_residuals']['0']['max_abs_cosine'], 0)
        self.assertEqual(report['sibling_residuals']['0']['rank'], 2)
        with self.assertRaisesRegex(AssertionError, 'Degenerate residual'):
            smoke.geometry(torch.tensor([[1., 0], [2., 0]]), [0], {0: [1]})


if __name__ == '__main__':
    unittest.main()
