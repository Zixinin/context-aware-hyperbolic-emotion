"""Target-environment CPU raw-weight audit before any expensive control training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from control_common import (ASSET_MANIFEST, C_RUN, MODEL_DIR, check_sources, config_for,
                            freeze, load_all_verified, package_versions, read, require,
                            require_runtime, sha, verify_c, verify_prediction_metrics, write)
from control_model import ControlBert, parameter_report
from preflight import EXPECTED_VERSIONS, verify_raw_encoder, verify_reference
from train import seed_everything
from model import LabelAwareBert


def audit(out, unit_proof, verify_only=False):
    require_runtime(True)
    sources = check_sources()
    require(read(unit_proof)["status"] == "EXP5A_CONTROL_UNIT_TESTS_PASS", "Control unit tests missing")
    require(read(unit_proof)["source_sha256"] == sources, "Unit source changed")
    versions = package_versions()
    require({k: versions[k] for k in EXPECTED_VERSIONS} == EXPECTED_VERSIONS, "Pinned environment differs")
    c = verify_c()
    datasets, manifest = load_all_verified()
    require(sha(ASSET_MANIFEST) == read(C_RUN / "environment.json")["dataset_manifest_sha256"], "C asset manifest changed")
    for name, digest in manifest["model"]["sha256"].items():
        require(sha(MODEL_DIR / name) == digest, f"Raw model asset changed: {name}")
    # Actual-environment initial encoder/head/RNG comparison with C.
    seed_everything(42)
    query = LabelAwareBert.from_pretrained(str(MODEL_DIR), config_for("full", "cls"))
    query_raw = verify_raw_encoder(query, manifest)
    initial_head = query.classifier_weight.detach().clone()
    initial_bias = query.classifier_bias.detach().clone()
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    del query
    parameters, raw = {}, {}
    for representation in ("cls", "mean"):
        seed_everything(42)
        model = ControlBert.from_pretrained(str(MODEL_DIR), config_for("full", representation))
        require(torch.equal(model.classifier_weight, initial_head) and torch.equal(model.classifier_bias, initial_bias), "C classifier initialization differs")
        require(torch.equal(torch.get_rng_state(), rng_cpu), "CPU RNG differs after initialization")
        require(all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng_cuda)), "CUDA RNG differs after initialization")
        raw[representation] = verify_raw_encoder(model, manifest)
        parameters[representation] = parameter_report(model)
        require(parameters[representation]["trainable_parameters"] == 107741212, "Unexpected control parameter count")
        del model
    require(parameters["cls"] == parameters["mean"], "Control parameter counts differ")
    require(c["parameters"]["exp5a_trainable_parameters"] - parameters["cls"]["trainable_parameters"] == 21504, "Unexpected C parameter increment")
    c_metrics, _, _ = verify_prediction_metrics(C_RUN, "test", datasets["test"])
    verify_prediction_metrics(C_RUN, "validation", datasets["validation"])
    report = dict(status="EXP5A_CONTROL_PREFLIGHT_PASS", source_sha256=sources,
                  unit_proof_sha256=sha(unit_proof), packages=versions, pip_freeze=freeze(),
                  c_reference=c, c_test_metrics=c_metrics, parameters=parameters,
                  raw_initialization=raw, c_raw_initialization=query_raw,
                  c_head_and_rng_parity=True, dataset_manifest_sha256=sha(ASSET_MANIFEST),
                  exp1_reference=verify_reference())
    if verify_only:
        require(read(out) == report, "Fresh audit differs from saved preflight")
    else:
        require(not Path(out).exists(), "Refusing to overwrite preflight")
        write(out, report)
    print(report["status"], flush=True)
    print({"parameters": parameters, "C_verified": str(C_RUN), "C_head_and_rng_parity": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unit-proof", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    audit(args.out, args.unit_proof, args.verify_only)
