"""Actual target preflight: frozen seed42 proof, raw weights and new-seed init parity."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multi_common import (AUTHORIZED, ASSET_MANIFEST, MODEL_DIR, check_sources, config_for, freeze,
                          load_all_verified, package_versions, read, require, require_runtime,
                          sha, verify_seed42, write)
from control_model import ControlBert, parameter_report
from model import LabelAwareBert
from preflight import EXPECTED_VERSIONS, verify_raw_encoder
from train import seed_everything


def audit(out, unit_proof, verify_only=False):
    require_runtime(True)
    source = check_sources()
    unit = read(unit_proof)
    require(unit["status"] == "EXP5A_MULTISEED_UNIT_TESTS_PASS" and unit["source_sha256"] == source, "Unit/source gate failed")
    versions = package_versions()
    require({k: versions[k] for k in EXPECTED_VERSIONS} == EXPECTED_VERSIONS, "Pinned environment changed")
    reference = verify_seed42()
    _, manifest = load_all_verified()
    for name, digest in manifest["model"]["sha256"].items():
        require(sha(MODEL_DIR / name) == digest, f"Raw BERT asset changed: {name}")
    parameters, initialization = {}, {}
    for seed in AUTHORIZED:
        seed_everything(seed)
        query = LabelAwareBert.from_pretrained(str(MODEL_DIR), config_for("full", "cls", seed))
        query_raw = verify_raw_encoder(query, manifest)
        head, bias = query.classifier_weight.detach().clone(), query.classifier_bias.detach().clone()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        del query
        for representation in ("cls", "mean"):
            seed_everything(seed)
            model = ControlBert.from_pretrained(str(MODEL_DIR), config_for("full", representation, seed))
            require(torch.equal(head, model.classifier_weight) and torch.equal(bias, model.classifier_bias), "Seedwise head init differs")
            require(torch.equal(cpu_rng, torch.get_rng_state()), "Seedwise CPU RNG differs")
            actual_cuda = torch.cuda.get_rng_state_all()
            require(len(actual_cuda) == len(cuda_rng) and all(torch.equal(a, b) for a, b in zip(actual_cuda, cuda_rng)), "Seedwise CUDA RNG differs")
            raw = verify_raw_encoder(model, manifest)
            count = parameter_report(model)
            require(count["trainable_parameters"] == 107741212 and count["query_parameters"] == 0, "Parameter count differs")
            if representation in parameters:
                require(parameters[representation] == count, "Counts differ across seeds")
            parameters[representation] = count
            initialization[f"{representation}_seed{seed}"] = dict(raw=raw, head_and_rng_parity=True, query_raw=query_raw)
            del model
    report = dict(status="EXP5A_MULTISEED_PREFLIGHT_PASS", source_sha256=source,
                  unit_proof_sha256=sha(unit_proof), packages=versions, pip_freeze=freeze(),
                  seed42_reference=reference, authorized_new_seeds=list(AUTHORIZED),
                  parameters=parameters, initialization=initialization,
                  dataset_manifest_sha256=sha(ASSET_MANIFEST))
    if verify_only:
        require(read(out) == report, "Fresh target audit differs from saved preflight")
    else:
        require(not Path(out).exists(), "Refusing to overwrite preflight")
        write(out, report)
    print(report["status"], flush=True)
    print(dict(authorized_new_seeds=list(AUTHORIZED), parameters=parameters,
               seed42_read_only_verified=reference["path"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unit-proof", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    audit(args.out, args.unit_proof, args.verify_only)
