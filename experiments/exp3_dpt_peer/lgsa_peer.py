"""Add peer loss to an existing LGSA instance, without editing upstream files."""
import inspect
import math
import types
import torch
from peer_contrastive import PeerHardNegativeContrastiveLoss


class LGSAWithPeer:
    """Non-Module adapter: does not add/reorder model or optimizer parameters."""
    def __init__(self, model, hierarchy, config):
        self.model = model
        self.original_forward = model.forward
        self.signature = inspect.signature(self.original_forward)
        self.config = dict(config)
        self.peer_lambda = float(config['peer_lambda'])
        if not math.isfinite(self.peer_lambda) or self.peer_lambda < 0:
            raise ValueError('peer_lambda must be finite and nonnegative')
        if hierarchy['depths'] != {0: [28, 29, 30, 31], 1: list(range(28))}:
            raise ValueError('Frozen two-level hierarchy required')
        parents = [hierarchy['parents'][i] for i in range(28)]
        self.loss_fn = PeerHardNegativeContrastiveLoss(parents, config['peer_temperature'], config['peer_top_k'])
        self.last = None
        self.records = []

    def attach(self):
        adapter = self
        def forward(_model, *args, **kwargs):
            return adapter.forward(*args, **kwargs)
        self.model.forward = types.MethodType(forward, self.model)
        return self

    def detach(self):
        self.model.forward = self.original_forward

    def forward(self, *args, **kwargs):
        # This fast path intentionally installs no hook and performs no peer ops.
        # It preserves original random draws and graph when lambda=0 or disabled.
        active = bool(self.config['peer_loss_enabled']) and self.peer_lambda != 0 and self.model.training
        bound = self.signature.bind(*args, **kwargs)
        labels = bound.arguments.get('labels')
        if not active or labels is None:
            self.last = None
            return self.original_forward(*args, **kwargs)
        ids = bound.arguments.get('input_ids')
        if ids is None or labels.shape != (len(ids), 64):
            raise ValueError('Peer adapter requires original [B,T] IDs and [B,64] targets')
        mask = ids == (self.model.get_input_embeddings().size - 1)
        if not bool((mask.sum(dim=1) == 2).all()):
            raise ValueError('Expected coarse and fine task MASKs, in that order')
        gold = labels.reshape(len(ids), 2, 32)[:, 1, :28]
        decoder = self.model.get_output_embeddings()
        captured = []
        def capture(_module, inputs):
            captured.append(inputs[0])
        handle = decoder.register_forward_pre_hook(capture)
        try:
            result = self.original_forward(*args, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1 or not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError('LGSA architecture/return contract changed')
        output, mlm, classification = result
        if output.loss is None:
            raise RuntimeError('Original supervised loss missing')
        b, h = len(ids), captured[0].shape[-1]
        anchor = captured[0][mask].reshape(b, 2, h)[:, 1, :]
        v = self.model.vocab_size
        scores = output.logits[mask].reshape(b, 2, -1)[:, 1, v:v+28] + self.model.multiclass_bias[:28]
        table = decoder.weight() if callable(decoder.weight) else decoder.weight
        keys = table[v:v+28]
        if keys.shape != (28, h):
            raise RuntimeError('Fine label decoder rows changed')
        loss, diagnostics = self.loss_fn(anchor, keys, scores, gold)
        original = output.loss
        output.loss = original + self.peer_lambda * loss
        self.last = {'original_forward_loss': float(original.detach()), 'peer_loss': float(loss.detach()),
                     'anchor_shape': list(anchor.shape), 'keys_shape': list(keys.shape),
                     'anchor_requires_grad': anchor.requires_grad, 'keys_requires_grad': keys.requires_grad,
                     **diagnostics}
        # Scalars only: never retain a batch autograd graph in diagnostics.
        self.records.append(self.last)
        return output, mlm, classification
