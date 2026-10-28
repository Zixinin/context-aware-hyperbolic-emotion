"""A0 forward unchanged; Formula A auxiliary-only extension, no distance BCE."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a1lib import CONFIG, require, build_hyperbolic, state_hash
from model27 import load_model as load_a0_model


def loss_components(logits, targets, z, decoder_logits, emotion_embeddings):
    b = len(logits)
    require(b > 0 and logits.shape == targets.shape == (b, 27), "Need 27-dimensional BCE targets")
    require(z.shape == (b, 100) and decoder_logits.shape == (b, 27) and emotion_embeddings.shape == (27, 100), "Auxiliary shapes invalid")
    tensors = [logits, targets, z, decoder_logits, emotion_embeddings]
    require(all(t.dtype == torch.float32 and t.device == logits.device and bool(torch.isfinite(t).all()) for t in tensors), "Invalid dtype/device/nonfinite loss inputs")
    require(not emotion_embeddings.requires_grad, "Label vectors must stay fixed")
    require(bool(((targets == 0) | (targets == 1)).all()) and bool((targets.sum(1) >= 1).all()), "Invalid multi-hot labels")
    require(bool((z.square().sum(-1) < 1).all()) and bool((emotion_embeddings.square().sum(-1) < 1).all()), "Outside Poincare ball; stop without changing the formula")
    q = targets / targets.sum(-1, keepdim=True)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    positive_ce = -(q * F.log_softmax(decoder_logits, dim=-1)).sum(-1)
    positive_dot = (q * (z @ emotion_embeddings.T)).sum(-1)
    auxiliary_per_sample = positive_ce - positive_dot
    auxiliary = auxiliary_per_sample.mean()
    total = bce + 0.1 * auxiliary
    require(bool(torch.isfinite(total)), "Nonfinite A1 objective")
    return dict(total=total, classification=bce, auxiliary=auxiliary,
                positive_ce=positive_ce, positive_dot=positive_dot,
                auxiliary_per_sample=auxiliary_per_sample, positive_count=targets.sum(-1))


class A1Model(nn.Module):
    def __init__(self, control, hyper, embeddings):
        super().__init__()
        self.control, self.hyper = control, hyper
        self.register_buffer("emotion_embeddings", embeddings.detach().clone())

    def forward(self, input_ids, attention_mask, labels=None):
        captured = {}
        def capture(_module, _args, output):
            captured["pooled"] = output.last_hidden_state[:, 0, :]
        # A read-only output hook on this NEW model instance reuses the exact
        # existing encoder/pooling/dropout/head forward with a single encoding.
        handle = self.control.encoder.register_forward_hook(capture)
        try:
            text = self.control(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        finally:
            handle.remove()
        z = self.hyper.encode(captured["pooled"])
        raw = self.hyper.decoder.decode(z)
        components = None if labels is None else loss_components(text["logits"], labels, z, raw, self.emotion_embeddings)
        return dict(text, loss=None if components is None else components["total"],
                    components=components, z=z, decoder_logits=raw)


def build(snapshot, names):
    # Caller seeds exactly as A0 immediately before this call.
    control, tokenizer, loading_info = load_a0_model(snapshot)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    py_before, np_before = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            hyper, embeddings, hyper_identity = build_hyperbolic(names)
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
    require(torch.equal(cpu_before, torch.get_rng_state()) and torch.equal(cuda_before, torch.cuda.get_rng_state()), "Auxiliary initialization perturbed shared RNG")
    require(random.getstate() == py_before, "Auxiliary initialization changed Python RNG")
    np_after = np.random.get_state()
    require(np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1]) and np_before[2:] == np_after[2:], "Auxiliary initialization changed NumPy RNG")
    model = A1Model(control, hyper, embeddings)
    return model, tokenizer, loading_info, hyper_identity


def parameter_groups(model):
    # Call the original optimizer builder without changing its grouping/rules.
    from model27 import optimizer_for
    return optimizer_for(model)


def gradient_report(params, require_nonzero=True):
    params = list(params)
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params if p.requires_grad), "Missing/nonfinite gradient")
    norm2 = sum(float(p.grad.detach().double().square().sum()) for p in params if p.grad is not None)
    if require_nonzero:
        require(np.isfinite(norm2) and norm2 > 0, "Required gradient is zero/nonfinite")
    return dict(norm=norm2 ** .5, parameter_tensors=len(params),
                nonzero_tensors=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params))


def module_gradients(model):
    return {k: gradient_report(params) for k, params in {
        "shared_encoder": model.control.encoder.parameters(),
        "classifier": [model.control.classifier_weight, model.control.classifier_bias],
        "hyperbolic_projection": model.hyper.encoder.parameters(),
        "auxiliary_decoder": model.hyper.decoder.parameters()}.items()}


def diagnostic_values(output):
    c = output["components"]
    scalar = lambda t: float(t.detach())
    bce, aux = scalar(c["classification"]), scalar(c["auxiliary"])
    require(bce > 0, "BCE denominator must be positive for ratio diagnostics")
    return dict(loss=scalar(c["total"]), bce=bce, auxiliary=aux, weighted_auxiliary=.1 * aux,
                weighted_auxiliary_to_bce=.1 * aux / bce,
                absolute_weighted_auxiliary_to_bce=abs(.1 * aux) / bce,
                positive_ce_mean=scalar(c["positive_ce"].mean()),
                positive_dot_mean=scalar(c["positive_dot"].mean()),
                max_ball_norm=scalar(output["z"].norm(dim=-1).max()))


def train_step(model, batch, optimizer, scheduler, detailed=False):
    ids = batch.pop("ids")
    x = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    output = model(**x)
    values = diagnostic_values(output)
    n = len(ids)
    (output["loss"] * n).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(n)
    grads = module_gradients(model) if detailed else None
    require(model.emotion_embeddings.grad is None, "Fixed embeddings received gradients")
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(values, batch_size=n, ids=ids, gradients=grads, unclipped_grad_norm=float(clipped),
                logits_shape=list(output["logits"].shape), targets_shape=list(x["labels"].shape))
