"""CPU unit tests only. This does not certify a real LGSA/GPU regression."""
import copy
from pathlib import Path
import sys
import types
import unittest
import json
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peer_contrastive import PeerHardNegativeContrastiveLoss
from lgsa_peer import LGSAWithPeer


class PeerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.loss = PeerHardNegativeContrastiveLoss([28, 28, 28, 29, 31])
        self.h = torch.randn(2, 8, requires_grad=True)
        self.e = torch.randn(5, 8, requires_grad=True)
        self.z = torch.tensor([[1., 8., 7., 99., 5.], [1., 2., 3., 4., 5.]], requires_grad=True)
        self.y = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 1]])

    def test_sibling_extraction(self):
        self.assertEqual(self.loss.siblings(0), [1, 2])
        self.assertEqual(self.loss.siblings(4), [])

    def test_self_never_negative(self):
        for _, y, negatives in self.loss.select(self.z, self.y):
            self.assertNotIn(y, negatives)

    def test_all_co_gt_excluded(self):
        self.assertEqual(self.loss.select(self.z, self.y), [(0, 0, [2]), (0, 1, [2])])

    def test_neutral_no_loss(self):
        loss, stats = self.loss(self.h[1:], self.e, self.z[1:], self.y[1:])
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(stats['number_of_peer_terms'], 0)

    def test_current_score_topk(self):
        y = torch.tensor([[1, 0, 0, 0, 0]])
        self.assertEqual(self.loss.select(self.z[:1], y), [(0, 0, [1])])
        changed = self.z[:1].detach().clone()
        changed[0, 2] = 10
        self.assertEqual(self.loss.select(changed, y), [(0, 0, [2])])
        many = PeerHardNegativeContrastiveLoss(self.loss.parent_of, top_k=10)
        self.assertEqual(many.select(self.z[:1], y), [(0, 0, [1, 2])])

    def test_finite_and_exact_objective(self):
        loss, stats = self.loss(self.h, self.e, self.z, self.y)
        expected = []
        for y in (0, 1):
            a = F.cosine_similarity(self.h[0:1], self.e[y:y+1])[0] / .1
            b = F.cosine_similarity(self.h[0:1], self.e[2:3])[0] / .1
            expected.append(F.softplus(b - a))
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        self.assertEqual(stats['number_of_peer_terms'], 2)
        self.assertEqual(stats['number_of_samples_with_peer_terms'], 1)

    def test_backward_and_detached_selection(self):
        loss, _ = self.loss(self.h, self.e, self.z, self.y)
        loss.backward()
        for tensor in (self.h, self.e):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.norm().item(), 0)
        self.assertIsNone(self.z.grad)

    def test_no_eligible_is_differentiable(self):
        y = torch.zeros_like(self.y)
        y[:, 4] = 1
        loss, stats = self.loss(self.h, self.e, self.z, y)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(self.h.grad, torch.zeros_like(self.h)))
        self.assertEqual(stats['number_of_peer_terms'], 0)

    def test_all_family_co_gt(self):
        y = torch.tensor([[1, 1, 1, 0, 0]])
        self.assertEqual(self.loss.select(self.z[:1], y), [])

    def test_bad_parameters_rejected(self):
        for tau in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PeerHardNegativeContrastiveLoss([1, 1], temperature=tau)

    def test_frozen_ids_and_parent_mapping(self):
        repo = ROOT.parents[1]
        frozen = json.loads((repo / 'experiments/exp2_1_multiseed/frozen/hierarchy.json').read_text())
        actual = json.loads((repo / 'experiments/exp2_lgsa/code/goemotions_adaptation/hierarchy.json').read_text())
        self.assertEqual(actual, frozen)
        self.assertEqual([r['fine_id'] for r in actual['labels']], list(range(28)))
        neutral = actual['labels'][27]
        self.assertEqual((neutral['fine_label'], neutral['coarse_id']), ('neutral', 31))


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.randn(67, 8), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.randn(67))

    def weight(self):
        return self.table

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class ToyLGSA(torch.nn.Module):
    """Exercise the real adapter contract with a tiny dropout classifier."""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(67, 8)
        self.transform = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.2))
        self.decoder = Decoder()
        self.multiclass_bias = torch.nn.Parameter(torch.zeros(32))
        self.vocab_size = 32

    def get_input_embeddings(self):
        return types.SimpleNamespace(size=67)

    def get_output_embeddings(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        z = self.decoder(self.transform(self.emb(input_ids)))
        loss = z.square().mean() if labels is not None else None
        return types.SimpleNamespace(loss=loss, logits=z), loss, loss


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ToyLGSA().train()
        self.ids = torch.tensor([[1, 66, 2, 66, 0], [3, 66, 4, 66, 0]])
        self.gold = torch.full((2, 2, 32), -100, dtype=torch.long)
        self.gold[:, 1, :28] = 0
        self.gold[:, 1, 0] = 1
        self.gold[1, 1, 27] = 1
        self.gold = self.gold.reshape(2, 64)
        self.hierarchy = {'depths': {0: [28, 29, 30, 31], 1: list(range(28))},
                          'parents': {i: 28 if i < 27 else 31 for i in range(28)}}
        self.cfg = {'peer_loss_enabled': True, 'peer_lambda': .5, 'peer_temperature': .1, 'peer_top_k': 1}

    def test_lambda_zero_exact_rng_outputs_gradients_optimizer(self):
        state = torch.get_rng_state()
        parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=1e-4)
        groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
        old = self.model(input_ids=self.ids, labels=self.gold)
        old[0].loss.backward()
        grads = [None if p.grad is None else p.grad.clone() for p in parameters]
        after = torch.get_rng_state()
        self.model.zero_grad(set_to_none=True)
        adapter = LGSAWithPeer(self.model, self.hierarchy, {**self.cfg, 'peer_lambda': 0}).attach()
        torch.set_rng_state(state)
        new = self.model(input_ids=self.ids, labels=self.gold)
        new[0].loss.backward()
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for a, b in ((old[0].logits, new[0].logits), (old[0].loss, new[0].loss)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual([id(p) for p in parameters], [id(p) for p in self.model.parameters()])
        self.assertEqual(groups, optimizer.state_dict()['param_groups'])
        for a, p in zip(grads, parameters):
            self.assertTrue(a is None and p.grad is None or a is not None and torch.equal(a, p.grad))
        self.assertIsNone(adapter.last)
        self.assertEqual(len(self.model.decoder._forward_pre_hooks), 0)

    def test_enabled_reuses_decoder_and_preserves_logits_rng(self):
        rng = torch.get_rng_state()
        old = self.model(input_ids=self.ids, labels=self.gold)
        after = torch.get_rng_state()
        adapter = LGSAWithPeer(self.model, self.hierarchy, self.cfg).attach()
        torch.set_rng_state(rng)
        new = self.model(input_ids=self.ids, labels=self.gold)
        self.assertTrue(torch.equal(old[0].logits, new[0].logits))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertGreater(adapter.last['peer_loss'], 0)
        self.assertFalse(adapter.last['keys_requires_grad'])
        new[0].loss.backward()
        self.assertGreater(self.model.transform[0].weight.grad.norm().item(), 0)
        self.assertEqual(adapter.last['anchor_shape'], [2, 8])
        self.assertEqual(len(self.model.decoder._forward_pre_hooks), 0)

    def test_evaluation_uses_original_path(self):
        adapter = LGSAWithPeer(self.model, self.hierarchy, self.cfg).attach()
        self.model.eval()
        result = self.model(input_ids=self.ids)
        self.assertIsNone(result[0].loss)
        self.assertIsNone(adapter.last)
        self.assertEqual(len(self.model.decoder._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main()
